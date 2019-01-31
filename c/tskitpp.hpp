/*
 * MIT License
 *
 * Copyright (c) 2019 Tskit Developers
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

/**
 * @file tskitpp.hpp
 * @brief Tskit C++ API.
 */
#ifndef TSKITPP_HPP
#define TSKITPP_HPP

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <memory>
#include <cassert>

#include <tskit.h>

/* JK: this is just getting stuff working. Probably the namespace should be declared
 * in each file or something? Looks like we can't have includes within a namespace
 * so just dropped all the includes in here to get it working. */

namespace tskit {

#include <tskitpp/tables.hpp>

}
#endif
