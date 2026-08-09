use pbrt_r4::paramdict::ParameterDictionary;
use std::sync::Arc;
use std::thread;

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn parameter_dictionary_is_send_sync() {
    assert_send_sync::<ParameterDictionary>();
}

#[test]
fn parameter_dictionary_supports_concurrent_reads() {
    let mut params = ParameterDictionary::new();
    params.add_string("string filename", "scene.pbrt");
    let params = Arc::new(params);

    let workers: Vec<_> = (0..8)
        .map(|_| {
            let params = Arc::clone(&params);
            thread::spawn(move || {
                assert_eq!(params.get_one_string("filename", "missing"), "scene.pbrt");
            })
        })
        .collect();

    for worker in workers {
        worker.join().unwrap();
    }
}

#[test]
fn parameter_dictionary_serializes_writes() {
    let mut params = ParameterDictionary::new();
    params.add_string("string filename", "scene.pbrt");
    let params = Arc::new(params);

    let workers: Vec<_> = (0..8)
        .map(|index| {
            let params = Arc::clone(&params);
            thread::spawn(move || {
                let mut values = params.get_strings_mut("filename").unwrap();
                values[0] = format!("scene-{index}.pbrt");
            })
        })
        .collect();

    for worker in workers {
        worker.join().unwrap();
    }

    assert!(params
        .get_one_string("filename", "missing")
        .starts_with("scene-"));
}

#[test]
fn cloned_parameter_dictionary_has_independent_values() {
    let mut original = ParameterDictionary::new();
    original.add_string("string filename", "original.pbrt");

    let mut cloned = original.clone();
    cloned.replace_one_string("string filename", "resolved/original.pbrt");

    assert_eq!(
        original.get_one_string("filename", "missing"),
        "original.pbrt"
    );
    assert_eq!(
        cloned.get_one_string("filename", "missing"),
        "resolved/original.pbrt"
    );
}
