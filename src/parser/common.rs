use crate::paramdict::wellknown_params;
use crate::paramdict::ParameterDictionary;
use crate::util::base::Float;
use crate::util::spectrum::*;
use nom::bytes;
use nom::character;
use nom::multi;
use nom::number;
use nom::sequence;
use nom::IResult;
use std::io::{Error, ErrorKind};

pub fn space0(s: &str) -> IResult<&str, &str> {
    return nom::combinator::recognize(multi::many0(space_or_comment))(s);
}

pub fn space1(s: &str) -> IResult<&str, &str> {
    return nom::combinator::recognize(multi::many1(space_or_comment))(s);
}

fn space_or_comment(s: &str) -> IResult<&str, &str> {
    return nom::branch::alt((
        character::complete::multispace1,
        sequence::preceded(
            character::complete::char('#'),
            bytes::complete::take_till(|c| c == '\n'),
        ),
    ))(s);
}

pub fn bool_literal(s: &str) -> IResult<&str, &str> {
    return nom::branch::alt((
        nom::bytes::complete::tag_no_case("true"),
        nom::bytes::complete::tag_no_case("false"),
    ))(s);
}

pub fn float_literal(s: &str) -> IResult<&str, &str> {
    return number::complete::recognize_float(s);
}

pub fn string_literal(s: &str) -> IResult<&str, &str> {
    return sequence::delimited(
        character::complete::char('"'),
        bytes::complete::take_until("\""),
        character::complete::char('"'),
    )(s);
}

pub fn parse_literal(s: &str) -> IResult<&str, &str> {
    return nom::branch::alt((float_literal, bool_literal, string_literal))(s);
}

pub fn parse_listed_literal(s: &str) -> IResult<&str, Vec<&str>> {
    let (s, r) = parse_literal(s)?;
    return Ok((s, vec![r]));
}

pub fn parse_list(s: &str) -> IResult<&str, Vec<&str>> {
    return sequence::delimited(
        character::complete::char('['),
        sequence::delimited(
            space0,
            multi::separated_list1(space1, parse_literal), //nom::branch::alt((number::complete::recognize_float, token))),
            space0,
        ),
        character::complete::char(']'),
    )(s);
}

pub fn get_param_type(s: &str) -> (&str, &str) {
    let ss: Vec<&str> = s.split_ascii_whitespace().collect();
    if ss.len() == 2 {
        return (ss[0], ss[1]);
    } else if ss.len() == 1 {
        if let Some(t) = wellknown_params::find_type_from_key(ss[0]) {
            return (t, ss[0]);
        } else {
            return ("", ss[0]);
        }
    } else {
        return ("", s);
    }
}

pub fn convert_bool(s: &str) -> Result<bool, std::str::ParseBoolError> {
    let s = String::from(s).to_lowercase();
    let s2: &str = &s;
    match s2 {
        "true" => return Ok(true),
        "false" => return Ok(false),
        "\"true\"" => return Ok(true),
        "\"false\"" => return Ok(false),
        _ => return s2.parse::<bool>(),
    }
}

pub fn parse_params(s: &str) -> IResult<&str, ParameterDictionary> {
    let (s, v) = multi::separated_list0(
        space1,
        nom::branch::permutation((
            sequence::terminated(string_literal, space1),
            nom::branch::alt((parse_list, parse_listed_literal)),
        )),
    )(s)?;
    let mut params = ParameterDictionary::new();
    for vv in &v {
        let org_key = vv.0;
        let (t, key) = get_param_type(org_key);
        let new_key = format!("{t} {key}");
        match t {
            "string" => {
                let s_values = &vv.1;
                params.add_strings(&new_key, &s_values);
            }
            "texture" => {
                let s_values = &vv.1;
                params.add_strings(&new_key, &s_values);
            }
            "spectrum" => {
                let s_values = &vv.1;
                let values: Result<Vec<Float>, _> =
                    s_values.iter().map(|s| s.parse::<Float>()).collect();
                if let Ok(values) = values {
                    let spectrum = if values.len() == 1 {
                        Some(Spectrum::from(values[0]))
                    } else if values.len() == 3 {
                        Some(Spectrum::from_rgb(&values, SpectrumType::Albedo))
                    } else if values.len() >= 2 && values.len() % 2 == 0 {
                        let mut lambda = Vec::with_capacity(values.len() / 2);
                        let mut sampled = Vec::with_capacity(values.len() / 2);
                        for pair in values.chunks_exact(2) {
                            lambda.push(pair[0]);
                            sampled.push(pair[1]);
                        }
                        params.add_sampled_spectrum(&new_key, &lambda, &sampled);
                        Some(Spectrum::from_sampled(&lambda, &sampled))
                    } else {
                        None
                    };
                    if let Some(spectrum) = spectrum {
                        params.add_spectrum(&new_key, &spectrum);
                    } else {
                        params.add_strings(&new_key, &s_values);
                    }
                } else {
                    params.add_strings(&new_key, &s_values);
                }
            }
            "bool" => {
                let s_values = &vv.1;
                let values: Result<Vec<bool>, _> =
                    s_values.iter().map(|s| convert_bool(s)).collect();
                let values = match values {
                    Ok(values) => values,
                    Err(_) => {
                        return Err(nom::Err::Failure(nom::error::Error::new(
                            s,
                            nom::error::ErrorKind::Fail,
                        )))
                    }
                };
                params.add_bools(&new_key, &values);
            }
            "integer" => {
                let s_values = &vv.1;
                let values: Vec<i32> = match s_values.iter().map(|s| s.parse::<i32>()).collect() {
                    Ok(values) => values,
                    Err(_) => {
                        return Err(nom::Err::Failure(nom::error::Error::new(
                            s,
                            nom::error::ErrorKind::Fail,
                        )))
                    }
                };
                params.add_ints(&new_key, &values);
            }
            "color" | "rgb" | "xyz" => {
                let s_values = &vv.1;
                let values: Vec<Float> = match s_values.iter().map(|s| s.parse::<Float>()).collect()
                {
                    Ok(values) => values,
                    Err(_) => {
                        return Err(nom::Err::Failure(nom::error::Error::new(
                            s,
                            nom::error::ErrorKind::Fail,
                        )))
                    }
                };
                match t {
                    "color" => params.add_color(&new_key, &values),
                    "rgb" => params.add_rgb(&new_key, &values),
                    "xyz" => params.add_xyz(&new_key, &values),
                    _ => {}
                }
            }
            "blackbody" => {
                let s_values = &vv.1;
                let values: Vec<Float> = match s_values.iter().map(|s| s.parse::<Float>()).collect()
                {
                    Ok(values) => values,
                    Err(_) => {
                        return Err(nom::Err::Failure(nom::error::Error::new(
                            s,
                            nom::error::ErrorKind::Fail,
                        )))
                    }
                };
                params.add_blackbody(&new_key, &values);
            }
            "point" | "point2" | "point3" | "point4" => {
                let s_values = &vv.1;
                let values: Vec<Float> = match s_values.iter().map(|s| s.parse::<Float>()).collect()
                {
                    Ok(values) => values,
                    Err(_) => {
                        return Err(nom::Err::Failure(nom::error::Error::new(
                            s,
                            nom::error::ErrorKind::Fail,
                        )))
                    }
                };
                params.add_point(&new_key, &values);
            }
            "vector" | "vector2" | "vector3" | "vector4" => {
                let s_values = &vv.1;
                let values: Vec<Float> = match s_values.iter().map(|s| s.parse::<Float>()).collect()
                {
                    Ok(values) => values,
                    Err(_) => {
                        return Err(nom::Err::Failure(nom::error::Error::new(
                            s,
                            nom::error::ErrorKind::Fail,
                        )))
                    }
                };
                params.add_point(&new_key, &values);
            }
            "normal" => {
                let s_values = &vv.1;
                let values: Vec<Float> = match s_values.iter().map(|s| s.parse::<Float>()).collect()
                {
                    Ok(values) => values,
                    Err(_) => {
                        return Err(nom::Err::Failure(nom::error::Error::new(
                            s,
                            nom::error::ErrorKind::Fail,
                        )))
                    }
                };
                params.add_point(&new_key, &values);
            }
            "float" => {
                let s_values = &vv.1;
                let values: Vec<Float> = match s_values.iter().map(|s| s.parse::<Float>()).collect()
                {
                    Ok(values) => values,
                    Err(_) => {
                        return Err(nom::Err::Failure(nom::error::Error::new(
                            s,
                            nom::error::ErrorKind::Fail,
                        )))
                    }
                };
                params.add_floats(&new_key, &values);
            }
            _ => {
                let s_values = &vv.1;
                let values: Vec<Float> = match s_values.iter().map(|s| s.parse::<Float>()).collect()
                {
                    Ok(values) => values,
                    Err(_) => {
                        return Err(nom::Err::Failure(nom::error::Error::new(
                            s,
                            nom::error::ErrorKind::Fail,
                        )))
                    }
                };
                params.add_floats(&new_key, &values);
            }
        }
    }
    let rest = s.trim_start();
    if v.is_empty() && rest.starts_with('"') {
        return Err(nom::Err::Failure(nom::error::Error::new(
            s,
            nom::error::ErrorKind::Fail,
        )));
    }
    if rest.starts_with('[') {
        return Err(nom::Err::Failure(nom::error::Error::new(
            s,
            nom::error::ErrorKind::Fail,
        )));
    }
    return Ok((s, params));
}
