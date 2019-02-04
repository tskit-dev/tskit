#ifndef TSKITPP_TABLE_COLLECTION_HPP
#define TSKITPP_TABLE_COLLECTION_HPP

#include <memory>
#include "NodeTable.hpp"

namespace tskit
{
    class TableCollection
    {
      private:
        class Impl;
        std::unique_ptr<Impl> pimpl;

      public:
        NodeTable nodes;

        explicit TableCollection(double sequence_length);
        ~TableCollection();

        // NOTE: copy constructor for illustration purposes.
        // Opinions vary on what it means to copy something
        // implemented with unique_ptr--not so unique, right?
        // But, it is a reasonable operation.
        TableCollection(const TableCollection& other);

        double get_sequence_length() const;

        bool is_equal(const TableCollection& other) const;
    };

    inline bool
    operator==(const TableCollection& lhs, const TableCollection& rhs)
    {
        return lhs.is_equal(rhs);
    }
} // namespace tskit

#endif

