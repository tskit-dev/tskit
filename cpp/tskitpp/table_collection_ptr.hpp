/**
 * @file table_collection_ptr.hpp
 * @brief C++ wrapper for tsk_table_collection_t
 */
#ifndef TSKCPP_TABLE_COLLECTION_PTR_HPP
#define TSKCPP_TABLE_COLLECTION_PTR_HPP

#include <tskit/tables.h>
#include <memory>

namespace tskit
{
    /*!
     *@brief unique_ptr wrapper around tsk_table_collection_t
     */
    using table_collection_ptr
        = std::unique_ptr<tsk_table_collection_t,
                          void (*)(tsk_table_collection_t*)>;

    table_collection_ptr
    make_table_collection_ptr()
    {
        return table_collection_ptr(new tsk_table_collection_t{},
                                    [](tsk_table_collection_t* tables) {
                                        tsk_table_collection_free(tables);
                                        delete tables;
                                    });
    }

    table_collection_ptr
    copy(const table_collection_ptr& self, tsk_flags_t options)
    {
        auto rv = make_table_collection_ptr();
        int res = tsk_table_collection_copy(self.get(), rv.get(), options);
        if (res == -1)
            {
                throw std::runtime_error(
                    "failure to copy table_collection_ptr");
            }
        return rv;
    }
} // namespace tskit

#endif

