#include <sstream>
#include <iostream>

#include <tskitpp.hpp>

using namespace std;

int
main()
{
    /* TODO: replace this with real test cases */

    tskit::NodeTable nodes(nullptr);
    nodes.add_row(0, 1.0);
    nodes.add_row(0, 2.0);
    std::cout << "Straight table: num_rows = " << nodes.get_num_rows() << endl;

    auto nodes_copy(nodes);
    assert(nodes == nodes_copy);

    tskit::TableCollection tables(10);
    std::cout << "Sequence length = " << tables.get_sequence_length() << endl;
    tables.nodes.add_row(0, 1.0);
    tables.nodes.add_row(0, 2.0);
    tables.nodes.add_row(0, 3.0);
    std::cout << "Via table collection: num_rows = " << tables.nodes.get_num_rows() << endl;

    // Copy construction
    auto tables_copy(tables);
    std::cout << "Sequence length of copy = " << tables_copy.get_sequence_length() << endl;
    std::cout << "Via table collection: num_rows in copy = " << tables_copy.nodes.get_num_rows() << endl;
    assert(tables == tables_copy);

    return 0;
}
