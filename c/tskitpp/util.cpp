#include <tskit/core.h>
#include <sstream>
#include <stdexcept>

namespace tskit
{
    namespace util
    {
        void
        check_error(int val)
        {
            if (val < 0)
                {
                    std::ostringstream o;
                    o << tsk_strerror(val);
                    throw std::runtime_error(o.str());
                }
        }
    } // namespace util
} // namespace tskit
