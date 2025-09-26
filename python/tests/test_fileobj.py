# MIT License
#
# Copyright (c) 2018-2023 Tskit Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Test cases for loading and dumping different types of files and streams
"""
import io
import multiprocessing
import os
import pathlib
import platform
import queue
import shutil
import socket
import socketserver
import tempfile
import traceback

import pytest
import tszip
from pytest import fixture

import tskit


IS_WINDOWS = platform.system() == "Windows"
IS_OSX = platform.system() == "Darwin"


class TestPath:
    @fixture
    def tempfile_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            yield f"{tmp_dir}/plain_path"

    def test_pathlib(self, ts_fixture, tempfile_name):
        ts_fixture.dump(tempfile_name)
        ts2 = tskit.load(tempfile_name)
        assert ts_fixture.tables == ts2.tables


class TestPathLib:
    @fixture
    def pathlib_tempfile(self):
        fd, path = tempfile.mkstemp(prefix="tskit_test_pathlib")
        os.close(fd)
        temp_file = pathlib.Path(path)
        yield temp_file
        temp_file.unlink()

    def test_pathlib(self, ts_fixture, pathlib_tempfile):
        ts_fixture.dump(pathlib_tempfile)
        ts2 = tskit.load(pathlib_tempfile)
        assert ts_fixture.tables == ts2.tables


class TestFileObj:
    @fixture
    def fileobj(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with open(f"{tmp_dir}/fileobj", "wb") as f:
                yield f

    def test_fileobj(self, ts_fixture, fileobj):
        ts_fixture.dump(fileobj)
        fileobj.close()
        ts2 = tskit.load(fileobj.name)
        assert ts_fixture.tables == ts2.tables

    def test_fileobj_multi(self, replicate_ts_fixture, fileobj):
        file_offsets = []
        for ts in replicate_ts_fixture:
            ts.dump(fileobj)
            file_offsets.append(fileobj.tell())
        fileobj.close()
        with open(fileobj.name, "rb") as f:
            for ts, file_offset in zip(replicate_ts_fixture, file_offsets):
                ts2 = tskit.load(f)
                file_offset2 = f.tell()
                assert ts.tables == ts2.tables
                assert file_offset == file_offset2


class TestFileObjRW:
    @fixture
    def fileobj(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pathlib.Path(f"{tmp_dir}/fileobj").touch()
            with open(f"{tmp_dir}/fileobj", "r+b") as f:
                yield f

    def test_fileobj(self, ts_fixture, fileobj):
        ts_fixture.dump(fileobj)
        fileobj.seek(0)
        ts2 = tskit.load(fileobj)
        assert ts_fixture.tables == ts2.tables

    def test_fileobj_multi(self, replicate_ts_fixture, fileobj):
        file_offsets = []
        for ts in replicate_ts_fixture:
            ts.dump(fileobj)
            file_offsets.append(fileobj.tell())
        fileobj.seek(0)
        for ts, file_offset in zip(replicate_ts_fixture, file_offsets):
            ts2 = tskit.load(fileobj)
            file_offset2 = fileobj.tell()
            assert ts.tables == ts2.tables
            assert file_offset == file_offset2


class TestFD:
    @fixture
    def fd(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pathlib.Path(f"{tmp_dir}/fd").touch()
            with open(f"{tmp_dir}/fd", "r+b") as f:
                yield f.fileno()

    def test_fd(self, ts_fixture, fd):
        ts_fixture.dump(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        ts2 = tskit.load(fd)
        assert ts_fixture.tables == ts2.tables

    def test_fd_multi(self, replicate_ts_fixture, fd):
        for ts in replicate_ts_fixture:
            ts.dump(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        for ts in replicate_ts_fixture:
            ts2 = tskit.load(fd)
            assert ts.tables == ts2.tables


class TestUnsupportedObjects:
    def test_string_io(self, ts_fixture):
        with pytest.raises(io.UnsupportedOperation, match=r"fileno"):
            ts_fixture.dump(io.StringIO())
        with pytest.raises(io.UnsupportedOperation, match=r"fileno"):
            tskit.load(io.StringIO())
        with pytest.raises(io.UnsupportedOperation, match=r"fileno"):
            ts_fixture.dump(io.BytesIO())
        with pytest.raises(io.UnsupportedOperation, match=r"fileno"):
            tskit.load(io.BytesIO())


def dump_to_stream(q_err, q_in, file_out):
    """
    Get tree sequences from `q_in` and ts.dump() them to `file_out`.
    Uncaught exceptions are placed onto the `q_err` queue.
    """
    try:
        with open(file_out, "wb") as f:
            while True:
                ts = q_in.get()
                if ts is None:
                    break
                ts.dump(f)
    except Exception as exc:
        tb = traceback.format_exc()
        q_err.put((exc, tb))


def load_from_stream(q_err, q_out, file_in):
    """
    tskit.load() tree sequences from `file_in` and put them onto `q_out`.
    Uncaught exceptions are placed onto the `q_err` queue.
    """
    try:
        with open(file_in, "rb") as f:
            while True:
                try:
                    ts = tskit.load(f)
                except EOFError:
                    break
                q_out.put(ts)
    except Exception as exc:
        tb = traceback.format_exc()
        q_err.put((exc, tb))


def stream(fifo, ts_list):
    """
    data -> q_in -> ts.dump(fifo) -> tskit.load(fifo) -> q_out -> data_out
    """
    q_err = multiprocessing.Queue()
    q_in = multiprocessing.Queue()
    q_out = multiprocessing.Queue()
    proc1 = multiprocessing.Process(target=dump_to_stream, args=(q_err, q_in, fifo))
    proc2 = multiprocessing.Process(target=load_from_stream, args=(q_err, q_out, fifo))
    proc1.start()
    proc2.start()
    for data in ts_list:
        q_in.put(data)

    q_in.put(None)  # signal the process that we're done
    proc1.join(timeout=3)
    if not q_err.empty():
        # re-raise the first child exception
        exc, tb = q_err.get()
        print(tb)
        raise exc
    if proc1.is_alive():
        # prevent hang if proc1 failed to join
        proc1.terminate()
        proc2.terminate()
        raise RuntimeError("proc1 (ts.dump) failed to join")
    ts_list_out = []
    for _ in ts_list:
        try:
            data_out = q_out.get(timeout=3)
        except queue.Empty:
            # terminate proc2 so we don't hang
            proc2.terminate()
            raise
        ts_list_out.append(data_out)
    proc2.join(timeout=3)
    if proc2.is_alive():
        # prevent hang if proc2 failed to join
        proc2.terminate()
        raise RuntimeError("proc2 (tskit.load) failed to join")

    assert len(ts_list) == len(ts_list_out)
    for ts, ts_out in zip(ts_list, ts_list_out):
        assert ts.tables == ts_out.tables


@pytest.mark.network
@pytest.mark.skipif(IS_WINDOWS, reason="No FIFOs on Windows")
@pytest.mark.skipif(IS_OSX, reason="FIFO flakey on OS X, issue #1170")
class TestFIFO:
    @fixture
    def fifo(self):
        temp_dir = tempfile.mkdtemp(prefix="tsk_test_streaming")
        temp_fifo = os.path.join(temp_dir, "fifo")
        os.mkfifo(temp_fifo)
        yield temp_fifo
        shutil.rmtree(temp_dir)

    def test_single_stream(self, fifo, ts_fixture):
        stream(fifo, [ts_fixture])

    def test_multi_stream(self, fifo, replicate_ts_fixture):
        stream(fifo, replicate_ts_fixture)


ADDRESS = ("localhost", 10009)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


class StoreEchoHandler(socketserver.BaseRequestHandler):
    def handle(self):
        while True:
            try:
                ts = tskit.load(self.request.fileno())
            except EOFError:
                break
            ts.dump(self.request.fileno())
        self.server.shutdown()


def server_process(q):
    server = Server(ADDRESS, StoreEchoHandler)
    # Tell the client (on the other end of the queue) that it's OK to open
    # a connection
    q.put(None)
    server.serve_forever()


@pytest.mark.network
@pytest.mark.skipif(IS_WINDOWS or IS_OSX, reason="Errors on systems without proper fds")
class TestSocket:
    @fixture
    def client_fd(self):
        # Use a queue to synchronise the startup of the server and the client.
        q = multiprocessing.Queue()
        _server_process = multiprocessing.Process(target=server_process, args=(q,))
        _server_process.start()
        q.get(timeout=3)
        client = socket.create_connection(ADDRESS)
        yield client.fileno()
        client.close()
        _server_process.join(timeout=3)

    def verify_stream(self, ts_list, client_fd):
        for ts in ts_list:
            ts.dump(client_fd)
            echo_ts = tskit.load(client_fd)
            assert ts.tables == echo_ts.tables

    def test_single_then_multi(self, ts_fixture, replicate_ts_fixture, client_fd):
        self.verify_stream([ts_fixture], client_fd)
        self.verify_stream(replicate_ts_fixture, client_fd)


def write_to_fifo(path, file_path):
    with open(path, "wb") as fifo:
        with open(file_path, "rb") as file:
            fifo.write(file.read())


def read_from_fifo(path, expected_exception, error_text, read_func):
    with open(path) as fifo:
        with pytest.raises(expected_exception, match=error_text):
            read_func(fifo)


def write_and_read_from_fifo(fifo_path, file_path, expected_exception, error_text):
    os.mkfifo(fifo_path)
    for read_func in [tskit.load, tskit.TableCollection.load]:
        read_process = multiprocessing.Process(
            target=read_from_fifo,
            args=(fifo_path, expected_exception, error_text, read_func),
        )
        read_process.start()
        write_process = multiprocessing.Process(
            target=write_to_fifo, args=(fifo_path, file_path)
        )
        write_process.start()
        write_process.join(timeout=3)
        read_process.join(timeout=3)


@pytest.mark.network
@pytest.mark.skipif(IS_WINDOWS, reason="No FIFOs on Windows")
class TestBadStream:
    def test_bad_stream(self, tmp_path):
        fifo_path = tmp_path / "fifo"
        bad_file_path = tmp_path / "bad_file"
        bad_file_path.write_bytes(b"bad data")
        write_and_read_from_fifo(
            fifo_path, bad_file_path, tskit.FileFormatError, "not in kastore format"
        )

    def test_legacy_stream(self, tmp_path):
        fifo_path = tmp_path / "fifo"
        legacy_file_path = os.path.join(
            os.path.dirname(__file__), "data", "hdf5-formats", "msprime-0.3.0_v2.0.hdf5"
        )
        write_and_read_from_fifo(
            fifo_path, legacy_file_path, tskit.FileFormatError, "not in kastore format"
        )

    def test_tszip_stream(self, tmp_path, ts_fixture):
        fifo_path = tmp_path / "fifo"
        zip_file_path = tmp_path / "tszip_file"
        tszip.compress(ts_fixture, zip_file_path)
        write_and_read_from_fifo(
            fifo_path, zip_file_path, tskit.FileFormatError, "not in kastore format"
        )
