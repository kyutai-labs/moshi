# Contributing to Moshi

## Pull Requests

Moshi is the implementation of a research paper.
Therefore, we do not plan on accepting many pull requests for new features.
However, we certainly welcome them for bug fixes.

1. Fork the repo and create your branch from `main`.
2. If you have changed APIs, update the documentation accordingly.
3. Ensure pre-commit hooks pass properly, in particular the linting and typing.
4. When changing the Rust code, run `cargo check`, `cargo clippy`, `cargo test`.
5. Accept the Contributor License Agreement (see after).

Note that in general, we will not accept refactoring of the code.


## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a Contributor License Agreement.

If you agree with the full CLA provided in the next paragraph, copy the following statement in your PR, changing your Github Handle:

> I, {your GitHub handle}, confirm that I have read and understood the terms of the CLA of Kyutai-labs, as outlined in the repository's CONTRIBUTING.md, and I agree to be bound by these terms.

The full CLA is provided as follows:

> I, {your GitHub handle}, hereby grant to Kyutai-labs a perpetual, worldwide, non-exclusive, royalty-free,
> irrevocable license to use, modify, distribute, and sublicense my Contributions.

> I understand and accept that Contributions are limited to modifications, improvements, or changes
> to the project’s source code submitted via pull requests. I accept that Kyutai-labs has full discretion to
> review, accept, reject, or request changes to any Contributions I submit, and that submitting
> a pull request does not guarantee its inclusion in the project.

> By submitting a Contribution, I grant Kyutai-labs a perpetual, worldwide license to use, modify,
> reproduce, distribute, and create derivative works based on my Contributions.
> I also agree to assign all patent rights for any inventions or improvements that arise from my Contributions,
> giving the Kyutai-labs full rights to file for and enforce patents.
> I understand that the Kyutai-labs may commercialize, relicense, or exploit the project and my Contributions without further notice or obligation to me.
> I confirm that my Contributions are original and that I have the legal right to grant this license.
> If my Contributions include third-party materials, I will ensure that I have the necessary permissions
> and will disclose this information. I accept that once my Contributions are integrated, they may be altered or removed at the Kyutai-labs’s discretion.

> I acknowledge that I am making these Contributions voluntarily and will not receive any compensation.
> Furthermore, I understand that all Contributions, including mine, are provided on an "as-is" basis, with no warranties.
> By submitting a pull request, I agree to be bound by these terms.

## Issues

Please submit issues on our Github repository.

## License

By contributing to Moshi, you agree that your contributions will be licensed
under the LICENSE-* files in the root directory of this source tree.
In particular, the rust code is licensed under APACHE, and the python code under MIT.
