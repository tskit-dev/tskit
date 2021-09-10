This roadmap is a rough plan of upcoming work. Contributions are welcome at anytime,
no matter where something is on this list.

## Next release
###[Python 0.3.8](https://github.com/tskit-dev/tskit/milestone/26) and [C 0.99.15](https://github.com/tskit-dev/tskit/milestone/28) 
- Complete IBD features [(project)](https://github.com/tskit-dev/tskit/projects/8)
- Large 64bit metadata columns (Python, already in C 0.99.14)
- Small additions cleanup and bugfixes (see milestone [issue list](https://github.com/tskit-dev/tskit/milestone/26))

## Nearterm goals

### [C API 1.0](https://github.com/tskit-dev/tskit/milestone/1)
- Document all structs and methods that will be stable and supported indefinitely 
- Remove legacy cruft
- Other clean up including sorting bugs and small features  (see milestone issue list)

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

### [Reference Sequence support](https://github.com/tskit-dev/tskit/issues/146)