use super::common::*;
use super::remove_comment::remove_comment;
use nom::bytes;
use nom::character;
use nom::multi;
use nom::sequence;
use nom::IResult;
use std::fs;
use std::io::Error;
use std::io::ErrorKind;
use std::io::Read;
use std::path::{Path, PathBuf};

pub fn read_file_with_include(path: &str) -> Result<String, Error> {
    let path = Path::new(path);
    if path.exists() {
        let path = path.canonicalize()?;
        let mut dirs = Vec::<PathBuf>::new();
        let parent = path.parent().ok_or(Error::from(ErrorKind::InvalidInput))?;
        dirs.push(PathBuf::from(parent));
        let mut ss = String::new();
        ss += &print_work_dir_begin(&dirs);
        ss += &read_file_with_include_core(path.as_path(), &mut dirs)?;
        ss += &print_work_dir_end();
        return Ok(ss);
    } else {
        return Err(Error::from(ErrorKind::NotFound));
    }
}

pub fn read_file_without_include(path: &str) -> Result<String, Error> {
    let path = Path::new(path);
    if path.exists() {
        return read_file_without_include_core(path);
    } else {
        return Err(Error::from(ErrorKind::NotFound));
    }
}

fn read_to_string(path: &Path) -> Result<String, Error> {
    let extent = path
        .extension()
        .ok_or(Error::from(ErrorKind::InvalidInput))?;
    let extent = extent.to_string_lossy().into_owned();
    if extent == "gz" {
        let f = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(f);
        let mut reader = flate2::read::GzDecoder::new(reader);
        let mut s = String::new();
        reader.read_to_string(&mut s)?;
        return Ok(s);
    } else {
        let s = fs::read_to_string(path)?;
        return Ok(s);
    }
}

fn read_file_with_include_core(path: &Path, dirs: &mut Vec<PathBuf>) -> Result<String, Error> {
    let s = read_to_string(path)?;
    return evaluate_include(&s, dirs);
}

pub fn read_file_without_include_core(path: &Path) -> Result<String, Error> {
    let s = read_to_string(path)?;
    return remove_comment_result(&s);
}

fn get_next_path(filename: &Path, dirs: &[PathBuf]) -> Option<PathBuf> {
    for d in dirs.iter().rev() {
        let dir = d;
        let path = dir.join(Path::new(filename));
        if path.exists() {
            return Some(path);
        }
    }
    return None;
}

fn print_work_dir_begin(dirs: &[PathBuf]) -> String {
    let path = &dirs[dirs.len() - 1];
    let path = path.as_path().to_string_lossy();
    return format!("WorkDirBegin \"{}\"\n", path);
}

fn print_work_dir_end() -> String {
    return "WorkDirEnd\n".to_string();
}

fn remove_comment_result(s: &str) -> Result<String, Error> {
    let r = remove_comment(s);
    match r {
        Ok((_, s)) => {
            return Ok(s);
        }
        Err(e) => {
            return Err(Error::new(ErrorKind::Other, e.to_string()));
        }
    }
}

fn parse_tokens(s: &str) -> Result<Vec<String>, Error> {
    let r = nom::combinator::all_consuming(nom::multi::many0(parse_one))(&s);
    match r {
        Ok((_, vs)) => {
            return Ok(vs);
        }
        Err(e) => {
            return Err(Error::new(ErrorKind::Other, e.to_string()));
        }
    }
}

fn evaluate_include(s: &str, dirs: &mut Vec<PathBuf>) -> Result<String, Error> {
    let s = remove_comment_result(s)?;
    let vs = parse_tokens(&s)?;

    let mut ss = String::new();
    //ss += &print_work_dir_begin(dirs);
    for s in vs {
        if s.starts_with("Include") {
            let vv: Vec<&str> = s.split('|').collect();
            let Some(filename_str) = vv.get(1) else {
                return Err(Error::from(ErrorKind::InvalidInput));
            };
            let filename = Path::new(filename_str);
            if let Some(next_path) = get_next_path(filename, dirs) {
                let Some(parent) = next_path.parent() else {
                    return Err(Error::from(ErrorKind::InvalidInput));
                };
                dirs.push(PathBuf::from(parent));
                ss += &print_work_dir_begin(dirs);

                let rss = read_file_with_include_core(next_path.as_path(), dirs)?;
                ss += &rss;
                if vv.len() > 2 {
                    for i in 2..vv.len() {
                        ss += &format!(" {}", vv[i]);
                    }
                    ss += "\n";
                }
                dirs.pop();
                ss += &print_work_dir_end();
            } else {
                return Err(Error::from(ErrorKind::NotFound));
            }
        } else {
            ss += &s;
        }
    }
    //ss += &print_work_dir_end();
    return Ok(ss);
}

fn parse_one(s: &str) -> IResult<&str, String> {
    return nom::branch::alt((
        parse_space1,
        parse_string_literal,
        parse_include,
        parse_token,
        parse_float,
        parse_any,
    ))(s);
}

fn parse_token(s: &str) -> IResult<&str, String> {
    let (s, (a, b)) = nom::branch::permutation((
        character::complete::alpha1,
        bytes::complete::take_while(|c: char| c.is_alphanumeric() || c == '_'),
    ))(s)?;
    return Ok((s, format!("{}{}", a, b)));
}

fn parse_float(s: &str) -> IResult<&str, String> {
    let (s, a) = nom::number::complete::recognize_float(s)?;
    return Ok((s, a.to_string()));
}

fn parse_any(s: &str) -> IResult<&str, String> {
    let (s, a) = character::complete::anychar(s)?;
    return Ok((s, a.to_string()));
}

fn parse_space1(s: &str) -> IResult<&str, String> {
    let (s, a) = character::complete::multispace1(s)?;
    return Ok((s, a.to_string()));
}

fn parse_string_literal(s: &str) -> IResult<&str, String> {
    let (s, a) = sequence::delimited(
        character::complete::char('"'),
        bytes::complete::take_until("\""),
        character::complete::char('"'),
    )(s)?;
    return Ok((s, format!("{}{}{}", "\"", a, "\"")));
}

pub fn parse_params(s: &str) -> IResult<&str, String> {
    let (s, v) = multi::separated_list0(
        space1,
        nom::branch::permutation((
            sequence::terminated(string_literal, space1),
            nom::branch::alt((parse_list, parse_listed_literal)),
        )),
    )(s)?;
    let mut ss = String::new();
    if v.len() > 0 {
        for (key, value) in v {
            let sv = value.join(" ");
            ss += &format!("|\"{}\" [{}]", key, sv);
        }
    }
    return Ok((s, ss));
}

fn parse_include(s: &str) -> IResult<&str, String> {
    let (s, (op, a, params)) = nom::branch::permutation((
        sequence::terminated(bytes::complete::tag("Include"), space1),
        sequence::terminated(string_literal, space0),
        parse_params,
    ))(s)?;
    let ss = format!("{}|{}{}", op, a, params);
    return Ok((s, ss));
}
