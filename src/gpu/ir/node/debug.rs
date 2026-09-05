use super::component::Component;
use super::node::{Node, NodeRef};
use super::shape::Shape;
use super::texture::TextureKind;
use crate::paramdict::ParameterDictionary;
use serde_json::{json, Map, Value};

pub fn node_ref_to_json(node: &NodeRef) -> Value {
    let node = node.read().unwrap_or_else(|poisoned| poisoned.into_inner());
    node_to_json(&node)
}

pub fn node_ref_to_json_string(node: &NodeRef) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&node_ref_to_json(node))
}

pub fn node_to_json(node: &Node) -> Value {
    json!({
        "name": node.name,
        "transform": node.transform.matrix,
        "components": node.components.iter().map(component_to_json).collect::<Vec<_>>(),
        "children": node.children.iter().map(node_ref_to_json).collect::<Vec<_>>(),
    })
}

pub fn node_to_json_string(node: &Node) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(&node_to_json(node))
}

fn component_to_json(component: &Component) -> Value {
    match component {
        Component::Scene(component) => json!({
            "type": "Scene",
            "textures": component.scene.textures.iter().map(|texture| {
                json!({
                    "name": texture.name,
                    "kind": texture_kind_name(texture.kind),
                    "params": params_to_json(&texture.params),
                    "transform": texture.transform.matrix,
                })
            }).collect::<Vec<_>>(),
        }),
        Component::Sampler(component) => named_params_to_json(
            "Sampler",
            &component.sampler.name,
            &component.sampler.params,
        ),
        Component::Integrator(component) => named_params_to_json(
            "Integrator",
            &component.integrator.name,
            &component.integrator.params,
        ),
        Component::Accelerator(component) => named_params_to_json(
            "Accelerator",
            &component.accelerator.name,
            &component.accelerator.params,
        ),
        Component::Filter(component) => {
            named_params_to_json("Filter", &component.filter.name, &component.filter.params)
        }
        Component::Camera(component) => json!({
            "type": "Camera",
            "medium": component.camera.medium,
            "params": params_to_json(&component.camera.params),
        }),
        Component::Film(component) => {
            named_params_to_json("Film", &component.film.name, &component.film.params)
        }
        Component::Output(component) => json!({
            "type": "Output",
            "filename": component.output.filename,
        }),
        Component::Shape(component) => {
            let mut value = shape_to_json(&component.shape);
            if let Some(object) = value.as_object_mut() {
                object.insert(
                    "reverse_orientation".to_string(),
                    json!(component.reverse_orientation),
                );
            }
            value
        }
        Component::Material(component) => json!({
            "type": "Material",
            "name": component.material.name,
            "kind": component.material.kind,
            "params": params_to_json(&component.material.params),
        }),
        Component::Light(component) => json!({
            "type": "Light",
            "name": component.light.name,
            "medium": component.light.medium,
            "params": params_to_json(&component.light.params),
            "transform": component.light.transform.matrix,
        }),
        Component::AreaLight(component) => json!({
            "type": "AreaLight",
            "name": component.area_light.name,
            "params": params_to_json(&component.area_light.params),
        }),
        Component::Medium(component) => json!({
            "type": "Medium",
            "name": component.medium.name,
            "params": params_to_json(&component.medium.params),
            "transform": component.medium.transform.matrix,
        }),
        Component::Instance(component) => json!({
            "type": "Instance",
            "target": node_name(&component.instance.target),
            "transform": component.instance.transform.matrix,
        }),
    }
}

fn named_params_to_json(kind: &str, name: &str, params: &ParameterDictionary) -> Value {
    json!({
        "type": kind,
        "name": name,
        "params": params_to_json(params),
    })
}

fn shape_to_json(shape: &Shape) -> Value {
    match shape {
        Shape::Sphere(sphere) => json!({
            "type": "Shape",
            "kind": "Sphere",
            "params": params_to_json(&sphere.params),
        }),
        Shape::Disk(disk) => json!({
            "type": "Shape",
            "kind": "Disk",
            "params": params_to_json(&disk.params),
        }),
        Shape::TriangleMesh(mesh) => json!({
            "type": "Shape",
            "kind": "TriangleMesh",
            "position_count": mesh.positions.len(),
            "index_count": mesh.indices.len(),
            "normal_count": mesh.normals.as_ref().map_or(0, Vec::len),
            "tangent_count": mesh.tangents.as_ref().map_or(0, Vec::len),
            "uv_count": mesh.uvs.as_ref().map_or(0, Vec::len),
        }),
    }
}

fn node_name(node: &NodeRef) -> String {
    let node = node.read().unwrap_or_else(|poisoned| poisoned.into_inner());
    node.name.clone()
}

fn texture_kind_name(kind: TextureKind) -> &'static str {
    match kind {
        TextureKind::Float => "Float",
        TextureKind::Spectrum => "Spectrum",
    }
}

fn params_to_json(params: &ParameterDictionary) -> Value {
    let mut output = Map::new();
    for key in params.get_keys() {
        let name = params.get_key_name(&key);
        let value = if let Some(values) = non_empty(params.get_bools(&name)) {
            json!(values)
        } else if let Some(values) = non_empty(params.get_ints(&name)) {
            json!(values)
        } else if let Some(values) = non_empty(params.get_floats(&name)) {
            json!(values)
        } else if let Some(values) = non_empty(params.get_strings(&name)) {
            json!(values)
        } else if let Some(values) = non_empty(params.get_points(&name)) {
            json!(values)
        } else if let Some(values) = non_empty(params.get_spectrums(&name)) {
            json!(values
                .iter()
                .map(|value| format!("{value:?}"))
                .collect::<Vec<_>>())
        } else if let Some(values) = non_empty(params.get_sampled_spectra(&name)) {
            json!(values
                .iter()
                .map(|value| format!("{value:?}"))
                .collect::<Vec<_>>())
        } else {
            Value::Null
        };
        output.insert(
            name,
            json!({
                "type": params.get_key_type(&key),
                "values": value,
            }),
        );
    }
    Value::Object(output)
}

fn non_empty<T>(values: Vec<T>) -> Option<Vec<T>> {
    (!values.is_empty()).then_some(values)
}
