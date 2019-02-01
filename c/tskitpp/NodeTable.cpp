#include "NodeTable.hpp"
#include "util.hpp"
#include <tskit/tables.h>

namespace tskit
{

    class NodeTable::NodeTableImpl
    {
      public:
        tsk_node_table_t* table;
        const bool allocated_locally;
        using node_table_ptr
            = std::unique_ptr<tsk_node_table_t, void (*)(tsk_node_table_t*)>;

        node_table_ptr
        make_empty_table()
        {
            node_table_ptr t(new tsk_node_table_t{},
                             [](tsk_node_table_t* nt) { delete nt; });
            if (t.get() == NULL)
                {
                    throw std::runtime_error("Out of memory");
                }
            return t;
        }

        tsk_node_table_t*
        allocate(tsk_node_table_t* the_table)
        {
            if (the_table != nullptr)
                {
                    return the_table;
                }

            node_table_ptr t(make_empty_table());
            int ret = tsk_node_table_init(t.get(), 0);
            // NOTE: check error is dangerous here.  If you don't
            // use a smart pointer above, you will get a leak.
            util::check_error(ret);
            return t.release();
        }

        tsk_node_table_t*
        copy_construct_details(const NodeTableImpl& other)
        {
            if (other.allocated_locally == false)
                {
                    // non-owning, so we share the pointer
                    // to data
                    return other.table;
                }
            node_table_ptr t(make_empty_table());
            int ret = tsk_node_table_copy(other.table, t.get(), 0);
            util::check_error(ret);
            return t.release();
        }

        NodeTableImpl(tsk_node_table_t* the_table)
            : table(allocate(the_table)), allocated_locally(the_table != table)
        {
        }

        NodeTableImpl(const NodeTableImpl& other)
            : table(copy_construct_details(other)),
              allocated_locally(other.allocated_locally)
        {
        }

        ~NodeTableImpl()
        {
            if (allocated_locally && table != NULL)
                {
                    tsk_node_table_free(table);
                    delete table;
                }
        }
    };

    NodeTable::NodeTable(tsk_node_table_t* the_table)
        : pimpl(new NodeTableImpl(the_table))
    {
    }

    NodeTable::NodeTable(const NodeTable& other)
        : pimpl(new NodeTableImpl(*other.pimpl))
    {
    }

    NodeTable::~NodeTable() = default;

    tsk_id_t NodeTable::add_row(tsk_flags_t flags,
                                double time) /* and more params */
    {
        tsk_id_t ret = tsk_node_table_add_row(pimpl->table, flags, time, TSK_NULL,
                                              TSK_NULL, NULL, 0);
        util::check_error(ret);
        return ret;
    }

    tsk_size_t
    NodeTable::get_num_rows(void) const
    {
        return pimpl->table->num_rows;
    }

    bool
    NodeTable::is_equal(const NodeTable& rhs) const
    // Convenience function allowing operator==
    // to be implemented without "friend" status.
    {
        return tsk_node_table_equals(this->pimpl->table, rhs.pimpl->table);
    }
    /* Etc */
} // namespace tskit
