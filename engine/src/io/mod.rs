use std::{fs, path::Path};

use serde::{de::DeserializeOwned, Serialize};

use crate::errors::Result;

pub mod simple;

pub fn load_json<T, P>(path: P) -> Result<T>
where
    T: DeserializeOwned,
    P: AsRef<Path>,
{
    let bytes = fs::read(path)?;
    let value = serde_json::from_slice(&bytes)?;
    Ok(value)
}

pub fn write_json<T, P>(path: P, value: &T) -> Result<()>
where
    T: Serialize,
    P: AsRef<Path>,
{
    let bytes = serde_json::to_vec_pretty(value)?;
    fs::write(path, bytes)?;
    Ok(())
}
