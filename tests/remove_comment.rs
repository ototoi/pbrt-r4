use pbrt_r4::parser::remove_comment::remove_comment;

#[test]
fn removes_comments_while_preserving_string_literals() {
    let (_, output) = remove_comment("\"aaa\" #1234\n aaa").unwrap();
    assert_eq!(output, "\"aaa\" \n aaa");
}
