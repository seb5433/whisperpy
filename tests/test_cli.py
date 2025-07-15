import subprocess


def test_main():
    assert subprocess.check_output(["whipy", "foo", "foobar"], text=True) == "foobar\n"
