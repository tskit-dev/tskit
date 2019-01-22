/* C++ wrapper experiement. */

#include <iostream>
#include <stdexcept>
#include <sstream>
#include <memory>
#include <cassert>

#include <tskit/tables.h>

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
        /* Etc */
};

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
};


int
main()
{
    NodeTable nodes(nullptr);
    nodes.add_row(0, 1.0);
    nodes.add_row(0, 2.0);
    std::cout << "Straight table: num_rows = " << nodes.get_num_rows() << endl;

    auto nodes_copy(nodes);
    assert(nodes.get_num_rows() == nodes_copy.get_num_rows());

    TableCollection tables(10);
    std::cout << "Sequence length = " << tables.get_sequence_length() << endl;
    tables.nodes.add_row(0, 1.0);
    tables.nodes.add_row(0, 2.0);
    tables.nodes.add_row(0, 3.0);
    std::cout << "Via table collection: num_rows = " << tables.nodes.get_num_rows() << endl;

    // Copy construction
    auto tables_copy(tables);
    std::cout << "Sequence length of copy = " << tables_copy.get_sequence_length() << endl;
    std::cout << "Via table collection: num_rows in copy = " << tables_copy.nodes.get_num_rows() << endl;

    return 0;

}
