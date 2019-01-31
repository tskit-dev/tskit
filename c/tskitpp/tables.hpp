/*
 * MIT License
 *
 * Copyright (c) 2019 Tskit Developers
 * Copyright (c) 2017-2018 University of Oxford
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
 * @file tables.hpp
 * @brief Tskit C++ Tables API.
 */
#ifndef TSKPP_TABLES_HPP
#define TSKPP_TABLES_HPP

/* JK: would like to include tskitpp/tables.hpp here but couldn't get it working. */
#include <tskitpp.hpp>

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <memory>
#include <cassert>

using namespace std;

void
check_error(int val)
{
    if (val < 0) {
        std::ostringstream o;
        o << tsk_strerror(val);
        throw std::runtime_error(o.str());
    }
}


class NodeTable
{

    private:
        tsk_node_table_t *table;
        const bool allocated_locally;
        using node_table_ptr = std::unique_ptr<tsk_node_table_t,void(*)(tsk_node_table_t*)>;

        node_table_ptr make_empty_table()
        {
            node_table_ptr t(new tsk_node_table_t{},[](tsk_node_table_t* nt){delete nt;});
            if (t.get() == NULL) {
                throw std::runtime_error("Out of memory");
            }
            return t;
        }

        tsk_node_table_t* allocate(tsk_node_table_t * the_table)
        {
            if (the_table != nullptr){ return the_table; }

            node_table_ptr t(make_empty_table());
            int ret = tsk_node_table_init(t.get(), 0);
            // NOTE: check error is dangerous here.  If you don't
            // use a smart pointer above, you will get a leak.
            check_error(ret);
            return t.release();
        }

        tsk_node_table_t*copy_construct_details(const NodeTable & other)
        {
            if(other.allocated_locally == false)
            {
                // non-owning, so we share the pointer
                // to data
                return other.table;
            }
            node_table_ptr t(make_empty_table());
            int ret = tsk_node_table_copy(other.table, t.get(), 0);
            check_error(ret);
            return t.release();
        }

    public:
        explicit NodeTable(tsk_node_table_t *the_table) : table(allocate(the_table)), allocated_locally(the_table != table)
        {
            std::cout << "table constructor " << endl;
        }

        NodeTable(const NodeTable & other) : table(copy_construct_details(other)), allocated_locally(other.allocated_locally)
        {
        }

        ~NodeTable()
        {
            if (allocated_locally && table != NULL) {
                tsk_node_table_free(table);
                delete table;
            }
        }

        tsk_id_t add_row(tsk_flags_t flags, double time) /* and more params */
        {
            tsk_id_t ret = tsk_node_table_add_row(table, flags, time, TSK_NULL,
                    TSK_NULL, NULL, 0);
            check_error(ret);
            return ret;
        }

        tsk_size_t get_num_rows(void) const
        {
            return table->num_rows;
        }

        bool is_equal(const NodeTable & rhs) const
        // Convenience function allowing operator==
        // to be implemented without "friend" status.
        {
            return tsk_node_table_equals(this->table, rhs.table);
        }
        /* Etc */
};

// NOTE: this will work when placed into namespace tskit
// due to "argument-depdendent lookup", or ADL
inline bool operator==(const NodeTable & lhs, const NodeTable & rhs)
{
    return lhs.is_equal(rhs);
}

class TableCollection
{

    private:
        using ptr = std::unique_ptr<tsk_table_collection_t, void(*)(tsk_table_collection_t*)>;
        ptr tables;

        ptr copy_table_collection(const ptr & other_tables)
        {
            ptr temp(new tsk_table_collection_t{},[](tsk_table_collection_t * t){delete t;});
            if(temp == nullptr) { throw std::runtime_error("Out of memory"); }
            int ret = tsk_table_collection_copy(other_tables.get(), temp.get(), 0);
            check_error(ret);
            return temp;
        }

    public:
        NodeTable nodes;

        explicit TableCollection(double sequence_length) : tables(new tsk_table_collection_t{},[](tsk_table_collection_t *t){delete t;}),
                 nodes(&tables->nodes)
        {
            if (tables == nullptr) {
                throw std::runtime_error("Out of memory");
            }
            int ret = tsk_table_collection_init(tables.get(), 0);
            // NOTE: without the smart pointer, this is a memory leak
            // waiting to happen.  The destructor is NOT called
            // if there is an exception from a constructor.
            check_error(ret);
            tables->sequence_length = sequence_length;
        }

        // NOTE: copy constructor for illustration purposes.
        // Opinions vary on what it means to copy something
        // implemented with unique_ptr--not so unique, right?
        // But, it is a reasonable operation.
        TableCollection(const TableCollection & other) :
            tables(copy_table_collection(other.tables)),
            nodes(&tables->nodes)
        {
            std::cout << "copy constructor " << '\n';
            if(tsk_table_collection_equals(tables.get(), other.tables.get()) == 0)
            {
                throw std::runtime_error("failure to copy table collection");
            }
        }

        ~TableCollection()
        {
            if (tables != nullptr) {
                tsk_table_collection_free(tables.get());
            }
        }

        double get_sequence_length() const
        {
            return tables->sequence_length;
        }

        bool is_equal(const TableCollection & other) const
        {
            return tsk_table_collection_equals(this->tables.get(), other.tables.get());
        }
};

inline bool operator==(const TableCollection & lhs, const TableCollection & rhs)
{
    return lhs.is_equal(rhs);
}

#endif
