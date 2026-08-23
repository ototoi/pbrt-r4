use crate::paramdict::wellknown_params;
use crate::parser::parsed_parameter::{ParsedParameter, ParsedParameterValues};
use nom::bytes;
use nom::character;
use nom::combinator::recognize;
use nom::multi;
use nom::number;
use nom::sequence;
use nom::IResult;

pub fn space0(s: &str) -> IResult<&str, &str> {
    return recognize(multi::many0_count(space_or_comment))(s);
}

pub fn space1(s: &str) -> IResult<&str, &str> {
    return recognize(multi::many1_count(space_or_comment))(s);
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
    let mut parts = s.split_ascii_whitespace();
    let Some(first) = parts.next() else {
        return ("", s);
    };
    let Some(second) = parts.next() else {
        return (
            wellknown_params::find_type_from_key(first).unwrap_or(""),
            first,
        );
    };
    if parts.next().is_none() {
        (first, second)
    } else {
        ("", s)
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

fn parsed_values<'a>(s: &'a str, parameter_type: &str) -> IResult<&'a str, ParsedParameterValues> {
    let (mut rest, in_array) = match nom::bytes::complete::tag::<_, _, nom::error::Error<_>>("[")(s)
    {
        Ok((rest, _)) => {
            let (rest, _) = space0(rest)?;
            (rest, true)
        }
        Err(_) => (s, false),
    };
    let mut values = ParsedParameterValues::new_for_type(parameter_type);
    loop {
        let (next, literal) = parse_literal(rest)?;
        values.push_literal(literal).map_err(|_| {
            nom::Err::Failure(nom::error::Error::new(rest, nom::error::ErrorKind::Fail))
        })?;
        rest = next;
        let (next, _) = match space1(rest) {
            Ok(value) => value,
            Err(_) => break,
        };
        rest = next;
        if in_array && rest.starts_with(']') {
            break;
        }
        if !in_array {
            break;
        }
    }
    let (rest_after_space, _) = space0(rest)?;
    rest = rest_after_space;
    if in_array {
        let (rest_after_close, _) = character::complete::char(']')(rest)?;
        Ok((rest_after_close, values))
    } else {
        Ok((rest, values))
    }
}

pub fn parse_params(s: &str) -> IResult<&str, Vec<ParsedParameter>> {
    let mut rest = s;
    let mut parameters = Vec::new();
    loop {
        let after_space = match space1(rest) {
            Ok((next, _)) => next,
            Err(_) => rest,
        };
        rest = after_space;
        if !rest.starts_with('"') {
            break;
        }
        let (next, declaration) = sequence::terminated(string_literal, space1)(rest)?;
        let (parameter_type, name) = get_param_type(declaration);
        let (next, values) = parsed_values(next, parameter_type)?;
        parameters.push(ParsedParameter {
            parameter_type: parameter_type.to_string(),
            name: name.to_string(),
            values,
        });
        rest = next;
    }
    if parameters.is_empty() && rest.trim_start().starts_with('"') {
        return Err(nom::Err::Failure(nom::error::Error::new(
            rest,
            nom::error::ErrorKind::Fail,
        )));
    }
    if rest.trim_start().starts_with('[') {
        return Err(nom::Err::Failure(nom::error::Error::new(
            rest,
            nom::error::ErrorKind::Fail,
        )));
    }
    Ok((rest, parameters))
}
