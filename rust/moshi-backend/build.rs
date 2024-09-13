// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

use anyhow::Result;
use vergen::EmitBuilder;

pub fn main() -> Result<()> {
    // NOTE: This will output everything, and requires all features enabled.
    // NOTE: See the EmitBuilder documentation for configuration options.
    EmitBuilder::builder().all_build().all_cargo().all_git().all_rustc().all_sysinfo().emit()?;
    Ok(())
}
