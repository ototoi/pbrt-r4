//! Owned, parser-side parameter values.
//!
//! This is the Rust counterpart of pbrt-v4's `ParsedParameter`.  It keeps
//! lexical type information separate from the renderer-facing
//! `ParameterDictionary` and lets a parse target take ownership of the
//! values without first collecting borrowed strings.

use crate::paramdict::{ParameterDictionary, SampledSpectrumParam};
use crate::util::base::Float;
use crate::util::spectrum::{Spectrum, SpectrumType};
use std::collections::HashSet;

#[derive(Debug, PartialEq)]
pub enum ParsedParameterValues {
    Bools(Vec<bool>),
    Ints(Vec<i32>),
    Floats(Vec<Float>),
    /// Values already stored in `ParameterDictionary`'s point storage.
    ///
    /// This is used only by the legacy dictionary-to-parser adapter. Parser
    /// input uses `Floats` and lets the dictionary classify the declaration.
    StoredPoints(Vec<Float>),
    Strings(Vec<String>),
    SpectrumTokens(Vec<ParsedSpectrumToken>),
    Spectrums(Vec<Spectrum>),
    SampledSpectrums(Vec<SampledSpectrumParam>),
}

#[derive(Debug, PartialEq)]
pub enum ParsedSpectrumToken {
    Float { value: Float, raw: String },
    String(String),
}

#[derive(Debug, PartialEq)]
pub struct ParsedParameter {
    pub parameter_type: String,
    pub name: String,
    pub values: ParsedParameterValues,
}

pub type ParsedParameterVector = Vec<ParsedParameter>;

/// Converts a legacy dictionary into the parser-owned representation.
///
/// The streaming parser does not use this adapter. It exists for targets such
/// as ToPly that still need to modify dictionary-shaped parameters.
pub fn parsed_parameters_from_dictionary(
    dictionary: &ParameterDictionary,
) -> ParsedParameterVector {
    let mut seen = HashSet::new();
    let mut result = Vec::new();
    for key in dictionary.get_keys() {
        if !seen.insert(key.clone()) {
            continue;
        }
        let mut parts = key.split_ascii_whitespace();
        let first = parts.next().unwrap_or_default();
        let rest = parts.collect::<Vec<_>>().join(" ");
        let (parameter_type, name) = if rest.is_empty() {
            (String::new(), first.to_string())
        } else {
            (first.to_string(), rest)
        };
        let storage_type = if parameter_type.is_empty() {
            if dictionary.get_strings_ref(&name).is_some() {
                "string"
            } else if dictionary.get_bools_ref(&name).is_some() {
                "bool"
            } else if dictionary.get_ints_ref(&name).is_some() {
                "integer"
            } else {
                match dictionary.get_key_type(&name).as_str() {
                    "point" | "point2" | "point3" | "point4" | "vector" | "vector2" | "vector3"
                    | "vector4" | "normal" | "color" | "rgb" | "xyz" => "point",
                    _ => "float",
                }
            }
        } else {
            parameter_type.as_str()
        };
        let values = match storage_type {
            "bool" => ParsedParameterValues::Bools(dictionary.get_bools(&name)),
            "integer" => ParsedParameterValues::Ints(dictionary.get_ints(&name)),
            "string" | "texture" => ParsedParameterValues::Strings(dictionary.get_strings(&name)),
            "spectrum" => {
                if dictionary.get_strings_ref(&name).is_some() {
                    ParsedParameterValues::Strings(dictionary.get_strings(&name))
                } else if dictionary.get_spectrums_ref(&name).is_some() {
                    ParsedParameterValues::Spectrums(dictionary.get_spectrums(&name))
                } else if dictionary.get_sampled_spectra_ref(&name).is_some() {
                    ParsedParameterValues::SampledSpectrums(dictionary.get_sampled_spectra(&name))
                } else {
                    ParsedParameterValues::Floats(dictionary.get_floats(&name))
                }
            }
            "point" | "point2" | "point3" | "point4" | "vector" | "vector2" | "vector3"
            | "vector4" | "normal" | "color" | "rgb" | "xyz" => {
                ParsedParameterValues::StoredPoints(dictionary.get_points(&name))
            }
            _ => ParsedParameterValues::Floats(dictionary.get_floats(&name)),
        };
        result.push(ParsedParameter {
            parameter_type,
            name,
            values,
        });
    }
    result
}

impl ParsedParameterValues {
    pub fn new_for_type(parameter_type: &str) -> Self {
        match parameter_type {
            "string" | "texture" => Self::Strings(Vec::new()),
            "spectrum" => Self::SpectrumTokens(Vec::new()),
            "bool" => Self::Bools(Vec::new()),
            "integer" => Self::Ints(Vec::new()),
            _ => Self::Floats(Vec::new()),
        }
    }

    pub fn push_literal(&mut self, literal: &str) -> Result<(), ()> {
        match self {
            Self::Bools(values) => values.push(
                literal
                    .parse::<bool>()
                    .or_else(|_| match literal.to_ascii_lowercase().as_str() {
                        "true" => Ok(true),
                        "false" => Ok(false),
                        _ => Err(()),
                    })
                    .map_err(|_| ())?,
            ),
            Self::Ints(values) => values.push(literal.parse::<i32>().map_err(|_| ())?),
            Self::Floats(values) => values.push(literal.parse::<Float>().map_err(|_| ())?),
            Self::StoredPoints(_) => return Err(()),
            Self::Strings(values) => values.push(literal.to_string()),
            Self::SpectrumTokens(values) => {
                if let Ok(value) = literal.parse::<Float>() {
                    values.push(ParsedSpectrumToken::Float {
                        value,
                        raw: literal.to_string(),
                    });
                } else {
                    values.push(ParsedSpectrumToken::String(literal.to_string()));
                }
            }
            Self::Spectrums(_) | Self::SampledSpectrums(_) => return Err(()),
        }
        Ok(())
    }
}

pub fn into_parameter_dictionary(parameters: ParsedParameterVector) -> ParameterDictionary {
    let mut dictionary = ParameterDictionary::new();
    for parameter in parameters {
        let parameter_type = parameter.parameter_type;
        let name = parameter.name;
        match parameter.values {
            ParsedParameterValues::Bools(values) => {
                dictionary.add_owned_bools_typed(&parameter_type, &name, values)
            }
            ParsedParameterValues::Ints(values) => {
                dictionary.add_owned_ints_typed(&parameter_type, &name, values)
            }
            ParsedParameterValues::Floats(values) => {
                dictionary.add_owned_floats_typed(&parameter_type, &name, values)
            }
            ParsedParameterValues::StoredPoints(values) => {
                dictionary.add_owned_points_typed(&parameter_type, &name, values)
            }
            ParsedParameterValues::Strings(values) => {
                dictionary.add_owned_strings_typed(&parameter_type, &name, values)
            }
            ParsedParameterValues::SpectrumTokens(tokens) => {
                let numeric = tokens
                    .iter()
                    .map(|token| match token {
                        ParsedSpectrumToken::Float { value, .. } => Ok(*value),
                        ParsedSpectrumToken::String(_) => Err(()),
                    })
                    .collect::<Result<Vec<_>, _>>();
                match numeric {
                    Ok(values) => {
                        if values.len() == 1 {
                            dictionary.add_owned_spectrums_typed(
                                &parameter_type,
                                &name,
                                vec![Spectrum::from(values[0])],
                            );
                        } else if values.len() == 3 {
                            dictionary.add_owned_spectrums_typed(
                                &parameter_type,
                                &name,
                                vec![Spectrum::from_rgb(&values, SpectrumType::Albedo)],
                            );
                        } else if values.len() >= 2 && values.len() % 2 == 0 {
                            let (lambda, sampled): (Vec<_>, Vec<_>) = values
                                .chunks_exact(2)
                                .map(|pair| (pair[0], pair[1]))
                                .unzip();
                            let spectrum = Spectrum::from_sampled(&lambda, &sampled);
                            dictionary.add_owned_sampled_spectra_typed(
                                &parameter_type,
                                &name,
                                vec![SampledSpectrumParam {
                                    lambda,
                                    values: sampled,
                                }],
                            );
                            dictionary.add_owned_spectrums_typed(
                                &parameter_type,
                                &name,
                                vec![spectrum],
                            );
                        } else {
                            dictionary.add_owned_strings_typed(
                                &parameter_type,
                                &name,
                                values.into_iter().map(|value| value.to_string()).collect(),
                            );
                        }
                    }
                    Err(()) => dictionary.add_owned_strings_typed(
                        &parameter_type,
                        &name,
                        tokens
                            .into_iter()
                            .map(|token| match token {
                                ParsedSpectrumToken::Float { raw, .. } => raw,
                                ParsedSpectrumToken::String(value) => value,
                            })
                            .collect(),
                    ),
                }
            }
            ParsedParameterValues::Spectrums(values) => {
                dictionary.add_owned_spectrums_typed(&parameter_type, &name, values)
            }
            ParsedParameterValues::SampledSpectrums(values) => {
                dictionary.add_owned_sampled_spectra_typed(&parameter_type, &name, values)
            }
        }
    }
    dictionary
}
