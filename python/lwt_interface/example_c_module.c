/*
 * MIT License
 *
 * Copyright (c) 2019-2020 Tskit Developers
 * Copyright (c) 2015-2018 University of Oxford
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
// Turn off clang-formatting for this file as turning off formatting
// for specific bits will make it more confusing.
// clang-format off

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <structmember.h>
#include <numpy/arrayobject.h>

#include "kastore.h"
#include "tskit.h"

#include "tskit_lwt_interface.h"

static PyMethodDef example_c_module_methods[] = {
    { NULL, NULL, 0, NULL } /* sentinel */
};

static struct PyModuleDef example_c_module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "example_c_module",
    .m_doc = "Example C module using the tskit LightweightTableCollection.",
    .m_size = -1,
    .m_methods = example_c_module_methods };

PyMODINIT_FUNC
PyInit_example_c_module(void)
{
    PyObject *module = PyModule_Create(&example_c_module);
    if (module == NULL) {
        return NULL;
    }
    import_array();
    if (register_lwt_class(module) != 0) {
        return NULL;
    }
    /* Put your own functions/class definitions here, as usual */

    return module;
}
