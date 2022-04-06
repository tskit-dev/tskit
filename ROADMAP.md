This roadmap is a rough plan of upcoming work. Contributions are welcome at anytime,
no matter where something is on this list.

## Upcoming releases
### [C API 1.0](https://github.com/tskit-dev/tskit/milestone/1)
- Reduce memory usage of loading and saving by making them zero in-memory copy.
- Document all structs and methods that will be stable and supported indefinitely. 
- Remove legacy cruft.
- Return genotypes as 32bit ints.
- Other clean up including sorting bugs and small features.  (see milestone issue list)


## [Python 0.5.0](https://github.com/tskit-dev/tskit/milestone/32)
- Python release to accompany C API 1.0.
- Random access to site genotypes

## [Python 0.5.1](https://github.com/tskit-dev/tskit/milestone/33)
- Python release to complete features needed for the tskit paper.
- Fast newick parsing

## [Python 0.5.2](https://github.com/tskit-dev/tskit/milestone/27)

## Medium term priorities

### [Metadata](https://github.com/tskit-dev/tskit/projects/7)
 - Improve user experience by adding convenience methods for updating, retrieving metadata and schemas
 - Codec performance
 - Standardised schemas

### [Stats API](https://github.com/tskit-dev/tskit/projects/6)
 - Added stats and options
 - Improve errors
 - (Fully deal with missing data)[https://github.com/tskit-dev/tskit/issues/287]
 - Time windowed statistics

### [Random Tree Access](https://github.com/tskit-dev/tskit/projects/5)
 - Build indexes to allow performant random tree access

