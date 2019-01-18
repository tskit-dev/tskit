/**
 * @file table_collection.hpp
 * @brief C++ wrapper for tsk_table_collection_t
 */
#ifndef TSK_CPP_TABLE_COLLECTION_HPP
#define TSK_CPP_TABLE_COLLECTION_HPP

#include <tskit/tables.h>
#include <memory>

namespace tskit
{
    /**
    @brief A wrapper around tsk_table_collection_t

    The type is move-only, which ensures safety because
    tsk_table_collection_t contains bare C pointers.  Thus, 
    disabling default copy operations on the C++ side is desired.

    TODO: need to mention how to copy
    */
    struct TableCollection
    {
        /** @brief Smart pointer holding the tsk_table_collection_t */
        std::unique_ptr<tsk_table_collection_t> tables;

        /** @brief Constructor
        TODO: should this be initalized or not?
        */
        TableCollection() : tables(new tsk_table_collection_t{}) {}

        inline tsk_table_collection_t* operator->() { return tables.get(); }

        inline const tsk_table_collection_t* operator->() const
        {
            return tables.get();
        }

        inline tsk_table_collection_t*
        get()
        {
            return tables.get();
        }

        inline const tsk_table_collection_t*
        get() const
        {
            return tables.get();
        }

        ~TableCollection() { tsk_table_collection_free(tables.get()); }
    };
} // namespace tskit

#endif

