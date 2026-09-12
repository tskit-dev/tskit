---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

```{currentmodule} tskit
```

(sec_debugging)=

# Debugging non-compliant tree sequences

If you're working on an application that writes out tree sequences
(or more properly, the underlying table collections),
you may find yourself in the situation where you're writing out files
that tskit cannot read because of violations of the {ref}`sec_data_model`.
How to see what's going on?

First, try loading the {class}`.TableCollection` directly (skipping
the convert-to-tree-sequence step that involves additional validation).
This will work if you have errors like a mutation that references a
non-extant node, for instance.
To do this, you simply run:
```{code-cell} python
import tskit
tables = tskit.TableCollection.load("data/basic_tree_seq.trees")
print(tables)
```
Then, the tables can be inspected.

Sometimes the error makes it so the tables cannot even be loaded
as a {class}`.TableCollection`: for instance, if something went wrong
with the {ref}`ragged columns<sec_encoding_ragged_columns>`.
Under the hood, data is stored on disk
with [kastore](https://github.com/tskit-dev/kastore),
so we can use kastore directly:
```{code-cell} python
import kastore
ka = kastore.load("data/basic_tree_seq.trees")
print(ka)
```
The kastore has (by design!) very minimal functionality:
we can ask what *keys* it has, and retrieve the associated values.
So, this one has these keys:
```{code-cell} python
list(ka.keys())
```
The names are self-explanatory.
For instance, we can look at the array of child nodes associated with each edge:
```{code-cell} python
ka.get("edges/child")
```

Good luck!
