#![allow(unsafe_op_in_unsafe_fn)]
#![allow(non_snake_case)]

use std::time::Duration;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

use crate::instances;

fn map_err(err: anyhow::Error) -> PyErr {
    PyRuntimeError::new_err(err.to_string())
}

#[pyfunction]
fn run_instance(py: Python<'_>, database_url: &str, payload: Bound<'_, PyBytes>) -> PyResult<i64> {
    let database_url = database_url.to_owned();
    let payload = payload.as_bytes().to_vec();
    py.allow_threads(move || instances::run_instance_blocking(&database_url, payload))
        .map_err(map_err)
}

#[pyfunction]
fn wait_for_instance(
    py: Python<'_>,
    database_url: &str,
    poll_interval_secs: Option<f64>,
) -> PyResult<Option<Py<PyBytes>>> {
    let database_url = database_url.to_owned();
    let interval = Duration::from_secs_f64(poll_interval_secs.unwrap_or(1.0).max(0.1));
    let result = py
        .allow_threads(move || instances::wait_for_instance_blocking(&database_url, interval))
        .map_err(map_err)?;
    Ok(result.map(|payload| PyBytes::new_bound(py, &payload).into()))
}

#[allow(non_snake_case)]
#[pymodule]
fn carabiner_worker__bridge(py: Python<'_>, module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(run_instance, module)?)?;
    module.add_function(wrap_pyfunction!(wait_for_instance, module)?)?;
    module.add(
        "__doc__",
        "Rust bridge for carabiner workflow orchestration",
    )?;
    if let Ok(sys) = py.import_bound("sys")
        && let Ok(modules) = sys.getattr("modules")
    {
        modules.set_item("carabiner_worker._bridge", module).ok();
    }
    Ok(())
}
