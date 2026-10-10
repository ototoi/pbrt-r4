use super::common::{parse_literal, space0};
use super::parse_target::ParseTarget;
use super::read_file::read_file_source;
use super::{evaluate_operation, parse_next_operation, parser_error_at, OPNode};
use crate::util::error::PbrtError;
use std::path::{Path, PathBuf};

struct InputFrame {
    path: PathBuf,
    source: String,
    offset: usize,
}

pub struct ParserSession<'a> {
    context: &'a mut dyn ParseTarget,
    frames: Vec<InputFrame>,
}

impl<'a> ParserSession<'a> {
    pub fn new(filename: &str, context: &'a mut dyn ParseTarget) -> Result<Self, PbrtError> {
        let path = Path::new(filename)
            .canonicalize()
            .map_err(|error| PbrtError::from(error).with_file(Path::new(filename)))?;
        let source =
            read_file_source(&path).map_err(|error| PbrtError::from(error).with_file(&path))?;
        let parent = path
            .parent()
            .ok_or_else(|| PbrtError::from("scene file has no parent directory"))?;
        context.work_dir_begin(&parent.to_string_lossy());

        Ok(Self {
            context,
            frames: vec![InputFrame {
                path,
                source,
                offset: 0,
            }],
        })
    }

    pub fn parse(mut self) -> Result<(), PbrtError> {
        let result = self.parse_frames();
        if result.is_err() {
            self.close_work_dirs();
        }
        result
    }

    fn parse_frames(&mut self) -> Result<(), PbrtError> {
        loop {
            let Some(frame) = self.frames.last() else {
                return Ok(());
            };
            if frame.offset == frame.source.len() {
                self.pop_frame();
                continue;
            }

            let Some((operation_offset, operation)) = self.next_operation()? else {
                self.pop_frame();
                continue;
            };

            if operation.name == "Include" {
                let filename = Self::include_filename(&operation).map_err(|error| {
                    self.operation_error(operation_offset, &operation.name, error)
                })?;
                if operation.params.is_none() {
                    return Err(self.operation_error(
                        operation_offset,
                        &operation.name,
                        PbrtError::error("Include requires parameters."),
                    ));
                }
                self.push_include(&filename).map_err(|error| {
                    self.operation_error(operation_offset, &operation.name, error)
                })?;
                continue;
            }

            let operation_name = operation.name.clone();
            if let Err(error) = evaluate_operation(operation, self.context) {
                let Some(frame) = self.frames.last() else {
                    return Err(Self::stack_empty_error());
                };
                return Err(parser_error_at(
                    &frame.source,
                    operation_offset,
                    &operation_name,
                    &error.to_string(),
                )
                .with_file(&frame.path));
            }
        }
    }

    fn next_operation(&mut self) -> Result<Option<(usize, OPNode)>, PbrtError> {
        let frame = self.frames.last().ok_or_else(Self::stack_empty_error)?;
        let input = &frame.source[frame.offset..];
        let parsed = parse_next_operation(&frame.source, input);
        let needs_parent = match &parsed {
            Ok(Some((remaining, _, operation))) => {
                remaining.is_empty() && operation.name != "Include"
            }
            Err(_) => true,
            Ok(None) => false,
        };
        if self.frames.len() == 1 || !needs_parent || has_unclosed_string(input) {
            let parsed = parsed.map_err(|error| self.add_current_file_context(error))?;
            let Some((remaining, operation_input, operation)) = parsed else {
                return Ok(None);
            };
            let next_offset = frame.source.len() - remaining.len();
            let operation_offset = frame.source.len() - operation_input.len();
            self.frames
                .last_mut()
                .ok_or_else(Self::stack_empty_error)?
                .offset = next_offset;
            return Ok(Some((operation_offset, operation)));
        }

        let (operation_input, _) =
            space0(input).map_err(|error| PbrtError::error(&error.to_string()))?;
        let operation_offset = frame.offset + input.len() - operation_input.len();
        let original_error = parsed.err();
        let mut joined = input.to_string();
        let last_index = self.frames.len() - 1;
        let mut spans = vec![(last_index, 0, joined.len())];
        // Read across EOF without closing the directive's work directory before
        // evaluation. This preserves source-relative paths on deferred entities.
        for index in (0..last_index).rev() {
            let parent = &self.frames[index];
            let tail = &parent.source[parent.offset..];
            let prefix_len = continuation_prefix_len(tail);
            let reaches_eof = prefix_len == tail.len();
            let prefix = &tail[..prefix_len];
            joined.push('\n');
            let start = joined.len();
            joined.push_str(prefix);
            spans.push((index, start, joined.len()));
            match parse_next_operation(&joined, &joined) {
                Ok(Some((remaining, _, operation))) => {
                    if remaining.is_empty() && reaches_eof && index != 0 {
                        continue;
                    }
                    let consumed = joined.len() - remaining.len();
                    for (frame_index, start, end) in spans {
                        self.frames[frame_index].offset += consumed.clamp(start, end) - start;
                    }
                    return Ok(Some((operation_offset, operation)));
                }
                Err(error) if index == 0 || !reaches_eof || has_unclosed_string(prefix) => {
                    return Err(self.operation_error(
                        operation_offset,
                        "<continuation>",
                        original_error.unwrap_or(error),
                    ));
                }
                _ => {}
            }
        }
        Err(Self::stack_empty_error())
    }

    fn include_filename(operation: &OPNode) -> Result<String, PbrtError> {
        let args = operation
            .args
            .as_ref()
            .ok_or_else(|| PbrtError::error("Include requires arguments."))?;
        args.get_strings("arg1")
            .into_iter()
            .next()
            .ok_or_else(|| PbrtError::error("Include requires a filename."))
    }

    fn push_include(&mut self, filename: &str) -> Result<(), PbrtError> {
        if self.frames.last().is_none() {
            return Err(Self::stack_empty_error());
        }

        let path = self
            .resolve_path(filename)
            .ok_or_else(|| PbrtError::from(format!("Include file not found: {filename}")))?;
        let path = path.canonicalize().map_err(PbrtError::from)?;
        if self.frames.iter().any(|frame| frame.path == path) {
            return Err(PbrtError::from(format!(
                "Include cycle detected at {}",
                path.display()
            )));
        }

        let source =
            read_file_source(&path).map_err(|error| PbrtError::from(error).with_file(&path))?;
        let parent = path
            .parent()
            .ok_or_else(|| PbrtError::from("included file has no parent directory"))
            .map_err(|error| error.with_file(&path))?;
        self.context.work_dir_begin(&parent.to_string_lossy());
        self.frames.push(InputFrame {
            path,
            source,
            offset: 0,
        });
        Ok(())
    }

    fn resolve_path(&self, filename: &str) -> Option<PathBuf> {
        let filename = Path::new(filename);
        if filename.is_absolute() && filename.exists() {
            return Some(filename.to_path_buf());
        }
        self.frames.iter().rev().find_map(|frame| {
            let parent = frame.path.parent()?;
            let path = parent.join(filename);
            path.exists().then_some(path)
        })
    }

    fn pop_frame(&mut self) {
        if self.frames.pop().is_some() {
            self.context.work_dir_end();
        }
    }

    fn close_work_dirs(&mut self) {
        while !self.frames.is_empty() {
            self.pop_frame();
        }
    }

    fn add_current_file_context(&self, error: PbrtError) -> PbrtError {
        match self.frames.last() {
            Some(frame) => error.with_file(&frame.path),
            None => error,
        }
    }

    fn operation_error(&self, offset: usize, operation: &str, error: PbrtError) -> PbrtError {
        match self.frames.last() {
            Some(frame) => parser_error_at(&frame.source, offset, operation, &error.to_string())
                .with_file(&frame.path),
            None => error,
        }
    }

    fn stack_empty_error() -> PbrtError {
        PbrtError::error("parser input stack unexpectedly empty")
    }
}

fn has_unclosed_string(input: &str) -> bool {
    let mut in_string = false;
    let mut in_comment = false;
    for character in input.chars() {
        if in_comment {
            in_comment = character != '\n';
        } else if character == '"' {
            in_string = !in_string;
        } else if character == '#' && !in_string {
            in_comment = true;
        }
    }
    in_string
}

// Stop before the next unquoted directive token. Literals and array brackets
// may continue a directive across file EOF; strings themselves may not.
fn continuation_prefix_len(input: &str) -> usize {
    let mut remaining = input;
    loop {
        let Ok((next, _)) = space0(remaining) else {
            return input.len() - remaining.len();
        };
        remaining = next;
        if remaining.is_empty() {
            return input.len();
        }
        if remaining.starts_with('[') || remaining.starts_with(']') {
            remaining = &remaining[1..];
        } else if let Ok((next, _)) = parse_literal(remaining) {
            remaining = next;
        } else if remaining.starts_with('"') {
            return input.len();
        } else {
            return input.len() - remaining.len();
        }
    }
}
