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

} // namespace tskit

#endif

