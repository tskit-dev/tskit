#ifndef TSKITPP_NODE_TABLE_HPP
#define TSKITPP_NODE_TABLE_HPP

#include <memory>
#include <tskit/tables.h>

namespace tskit
{

    class NodeTable
    {
      private:
        class NodeTableImpl;
        std::unique_ptr<NodeTableImpl> pimpl;

      public:
        explicit NodeTable(tsk_node_table_t* the_table);
        NodeTable(const NodeTable& other);
        ~NodeTable();

        tsk_id_t add_row(tsk_flags_t flags, double time); /* and more params */

        tsk_size_t get_num_rows(void) const;

        bool is_equal(const NodeTable& rhs) const;
        /* Etc */
    };

    // NOTE: this will work when placed into namespace tskit
    // due to "argument-depdendent lookup", or ADL
    inline bool
    operator==(const NodeTable& lhs, const NodeTable& rhs)
    {
        return lhs.is_equal(rhs);
    }
} // namespace tskit

#endif
