//! Input and CPU reference semantics are application-owned; this module knows no GPU API.
use std::io::{self, Read};
use std::time::Instant;

#[derive(Debug, Clone)]
pub(crate) struct Frame {
    pub(crate) id: usize,
    pub(crate) q: u32,
    pub(crate) pixels: Vec<u32>,
    pub(crate) receipt: Instant,
    pub(crate) external_capture_us: Option<u64>,
}

pub(crate) struct ExpectedFrame {
    pub(crate) frame: Frame,
    pub(crate) mask: Vec<u32>,
}

pub(crate) struct Reference {
    background: Vec<u32>,
}

impl Reference {
    pub(crate) fn new(pixels: usize) -> Self {
        Self {
            background: vec![0; pixels],
        }
    }

    pub(crate) fn accept(&mut self, frame: Frame) -> Result<ExpectedFrame, String> {
        if frame.pixels.len() != self.background.len() || !matches!(frame.q, 2 | 4) {
            return Err(format!(
                "invalid dimensions or adaptation in frame {}",
                frame.id
            ));
        }
        let mut mask = Vec::with_capacity(frame.pixels.len());
        for (&sample, old) in frame.pixels.iter().zip(&mut self.background) {
            if sample > 255 {
                return Err(format!("invalid grayscale value in frame {}", frame.id));
            }
            mask.push(if sample.abs_diff(*old) >= 32 { 255 } else { 0 });
            *old = ((frame.q - 1) * *old + sample) / frame.q;
        }
        Ok(ExpectedFrame { frame, mask })
    }

    pub(crate) fn background(&self) -> &[u32] {
        &self.background
    }
}

pub(crate) fn generated(id: usize, width: u32, height: u32) -> Frame {
    let mut pixels = Vec::with_capacity((width as usize) * (height as usize));
    let side = (width.min(height) / 8).max(1);
    let max_left = width.saturating_sub(side).max(1);
    let left = u32::try_from(id).unwrap_or(u32::MAX).wrapping_mul(3) % max_left;
    for y in 0..height {
        for x in 0..width {
            let level = 20 + ((x / 16 + y / 16) % 4) * 38;
            let moving = x >= left && x < left + side && y >= height / 3 && y < height / 3 + side;
            pixels.push(if moving {
                240
            } else {
                (level + ((id / 8) as u32 % 9)).min(255)
            });
        }
    }
    Frame {
        id,
        q: if id % 2 == 0 { 2 } else { 4 },
        pixels,
        receipt: Instant::now(),
        external_capture_us: None,
    }
}

/// Framed stdin: little-endian capture Unix microseconds (u64), q (u32), then width*height
/// grayscale bytes. EOF *between* packets ends the stream; partial packets are errors.
/// Capture clock is metadata, never silently assumed synchronized with consumer Instant.
pub(crate) fn read_stdin_frame(
    source: &mut impl Read,
    id: usize,
    pixel_count: usize,
) -> io::Result<Option<Frame>> {
    let mut header = [0_u8; 12];
    let mut first = [0_u8; 1];
    if source.read(&mut first)? == 0 {
        return Ok(None);
    }
    header[0] = first[0];
    source.read_exact(&mut header[1..])?;
    let mut grayscale = vec![0_u8; pixel_count];
    source.read_exact(&mut grayscale)?;
    let external_capture_us = u64::from_le_bytes(header[..8].try_into().expect("fixed header"));
    let q = u32::from_le_bytes(header[8..].try_into().expect("fixed header"));
    if !matches!(q, 2 | 4) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "q must be 2 or 4",
        ));
    }
    Ok(Some(Frame {
        id,
        q,
        pixels: grayscale.into_iter().map(u32::from).collect(),
        receipt: Instant::now(),
        external_capture_us: Some(external_capture_us),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_reference_uses_previous_background_and_exact_threshold() {
        let mut reference = Reference::new(2);
        let frame = Frame {
            id: 1,
            q: 4,
            pixels: vec![32, 31],
            receipt: Instant::now(),
            external_capture_us: None,
        };
        assert_eq!(reference.accept(frame).unwrap().mask, vec![255, 0]);
        assert_eq!(reference.background(), &[8, 7]);
        let frame = Frame {
            id: 2,
            q: 2,
            pixels: vec![8, 255],
            receipt: Instant::now(),
            external_capture_us: None,
        };
        assert_eq!(reference.accept(frame).unwrap().mask, vec![0, 255]);
        assert_eq!(reference.background(), &[8, 131]);
    }

    #[test]
    fn truncated_stdin_frame_fails_instead_of_accepting_partial_input() {
        let mut bytes: &[u8] = &[0_u8; 15];
        assert_eq!(
            read_stdin_frame(&mut bytes, 0, 4).unwrap_err().kind(),
            io::ErrorKind::UnexpectedEof
        );
    }
}
