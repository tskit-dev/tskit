# Abstract

The purpose of this document is to formalize the governance process used by the
tskit-dev community for core software projects, to clarify who has what responsibilities,
and how decisions are made on changes to those responsibilities.

tskit-dev is an open and inclusive community. Anyone with an interest in the
software and science is welcome to contribute discussion, and propose changes to code,
documentation or website content, as long as they follow the
code of conduct](https://github.com/tskit-dev/.github/blob/main/CODE_OF_CONDUCT.md).

# Scope

The governance model described in this document applies to the core
repositories of the tskit software ecosystem. These are:

- [tskit - C and Python API](http://github.com/tskit-dev/tskit)
- [msprime - coalescent simulator](http://github.com/tskit-dev/msprime)
- [tszip - compression library for tskit](http://github.com/tskit-dev/tszip)
- [tskit-site - website for the ecosystem](http://github.com/tskit-dev/tskit-site)
- [kastore - key-value store](http://github.com/tskit-dev/kastore)
- [administrative](http://github.com/tskit-dev/administrative)
- [.github - common config](http://github.com/tskit-dev/.github)


# Roles

## Contributors

Anyone is welcome to contribute to the tskit-dev project by, for example,

- proposing, discussing, or reviewing a change to the code, documentation, or specification
  via a GitHub pull request to the above repositories;
- reporting a GitHub issue or starting a discussion on the above repositories;

Contributors must abide by the [CODE OF CONDUCT](https://github.com/tskit-dev/.github/blob/main/CODE_OF_CONDUCT.md).


## Maintainers

Maintainers are those who have the "Maintain" role on a given repository. They are able
to merge pull requests, manage issues, and perform other administrative tasks on
that repository. Maintainers are added and removed by a decision of the tskit
Steering Council (TSC). This role is defined on a per-repository basis.


## tskit Steering Council

The tskit Steering Council (TSC) has the following responsibilities:

- Management of the tskit-dev GitHub organization, including the addition and removal of members.
- Administration of the repositories listed above.
- Addition and removal of maintainers to those repositories.
- Addition and removal of members of the tskit-dev Slack.
- Addition and removal of repositories from this governance model.
- Approval of changes to this governance model.
- Resolution of disputes between maintainers and contributors.
- Response to emails on the admin@tskit.dev address.
- Management of release artifacts on package indexes such as PyPI, conda-forge.

The steering council consists of a small number of people. This should always be an odd number to ensure a simple majority vote outcome is always possible. Currently this is:

* [Jerome Kelleher](https://github.com/jeromekelleher)

* [Peter Ralph](https://github.com/petrelharp)

* [Yan Wong](https://github.com/hyanwong)


TSC members may be removed by consensus of the remaining TSC members, or may resign voluntarily. New TSC members are added by unanimous consent of existing TSC members.

# Decision Making Process

Decisions about the future of the project are made through discussion with all
members of the community. All non-sensitive project management discussion takes
place on the issue trackers of the appropriate repositories.
Where possible decisions and discussions of the steering council should be documented as issues on the [administrative](https://github.com/tskit-dev/administrative) repository. Sensitive discussion may occur via email to admin@tskit.dev.

tskit uses a "consensus-seeking" process for making decisions. The group tries to
find a resolution that has no open objections among maintainers and the TSC. All
are expected to distinguish between fundamental objections to a proposal and minor perceived flaws that they can live with and not hold up the decision-making process for the latter. If no option can be found without objections, the decision is escalated to the TSC, which will use consensus to come to a resolution. In the unlikely event that consensus cannot be reached within the TSC, the proposal will move forward if it has the support of a simple majority of the TSC.

