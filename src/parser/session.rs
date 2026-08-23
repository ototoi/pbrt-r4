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

            let parsed = {
                let Some(frame) = self.frames.last() else {
                    return Err(Self::stack_empty_error());
                };
                let input = &frame.source[frame.offset..];
                parse_next_operation(&frame.source, input).map(|result| {
                    result.map(|(remaining, operation_input, operation)| {
                        (
                            frame.source.len() - remaining.len(),
                            frame.source.len() - operation_input.len(),
                            operation,
                        )
                    })
                })
            };

            let parsed = match parsed {
                Ok(parsed) => parsed,
                Err(error) => {
                    return Err(self.add_current_file_context(error));
                }
            };
            let Some((next_offset, operation_offset, operation)) = parsed else {
                self.pop_frame();
                continue;
            };

            let Some(frame) = self.frames.last_mut() else {
                return Err(Self::stack_empty_error());
            };
            frame.offset = next_offset;

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
