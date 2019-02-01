#include "TableCollection.hpp"
#include "util.hpp"
#include <stdexcept>

namespace tskit
{
    class TableCollection::TableCollectionImpl
    {
      public:
        using ptr = std::unique_ptr<tsk_table_collection_t,
                                    void (*)(tsk_table_collection_t*)>;
        ptr tables;

        TableCollectionImpl(const double sequence_length)
            : tables(new tsk_table_collection_t{},
                     [](tsk_table_collection_t* t) { delete t; })
        {
            int ret = tsk_table_collection_init(tables.get(), 0);
            // NOTE: without the smart pointer, this is a memory leak
            // waiting to happen.  The destructor is NOT called
            // if there is an exception from a constructor.
            util::check_error(ret);
            tables->sequence_length = sequence_length;
        }

        TableCollectionImpl(const TableCollectionImpl& other)
            : tables(copy_table_collection(other.tables))
        {
            if (tsk_table_collection_equals(tables.get(), other.tables.get())
                == 0)
                {
                    throw std::runtime_error(
                        "failure to copy table collection");
                }
        }

        ptr
        copy_table_collection(const ptr& other_tables)
        {
            ptr temp(new tsk_table_collection_t{},
                     [](tsk_table_collection_t* t) { delete t; });
            if (temp == nullptr)
                {
                    throw std::runtime_error("Out of memory");
                }
            int ret
                = tsk_table_collection_copy(other_tables.get(), temp.get(), 0);
            util::check_error(ret);
            return temp;
        }

        ~TableCollectionImpl()
        {
            if (tables != nullptr)
                {
                    tsk_table_collection_free(tables.get());
                }
        }
    };

    TableCollection::TableCollection(double sequence_length)
        : pimpl(new TableCollectionImpl(sequence_length)),
          nodes(&pimpl->tables->nodes)
    {
    }

    // NOTE: copy constructor for illustration purposes.
    // Opinions vary on what it means to copy something
    // implemented with unique_ptr--not so unique, right?
    // But, it is a reasonable operation.
    TableCollection::TableCollection(const TableCollection& other)
        : pimpl(new TableCollectionImpl(*other.pimpl.get())),
          nodes(&pimpl->tables->nodes)
    {
    }

    TableCollection::~TableCollection() = default;

    double
    TableCollection::get_sequence_length() const
    {
        return pimpl->tables->sequence_length;
    }

    bool
    TableCollection::is_equal(const TableCollection& other) const
    {
        return tsk_table_collection_equals(this->pimpl->tables.get(),
                                           other.pimpl->tables.get());
    }
} // namespace tskit
