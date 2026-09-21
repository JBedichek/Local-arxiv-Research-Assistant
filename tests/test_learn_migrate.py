from lara.learn import migrate, store


def make(root, name, text="x"):
    (root / name).mkdir(parents=True)
    (root / name / "course.json").write_text(text)


def test_copies_courses_and_leaves_the_source_alone(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make(src, "a-1"); make(src, "_shared")
    out = migrate.copy_courses(src, dst)
    assert out == {"copied": ["_shared", "a-1"], "skipped": []}
    assert (dst / "a-1" / "course.json").read_text() == "x" and (src / "a-1" / "course.json").exists()


def test_never_overwrites_what_the_destination_already_has(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    make(src, "a-1", "from source"); make(dst, "a-1", "already here")
    out = migrate.copy_courses(src, dst)
    assert out == {"copied": [], "skipped": ["a-1"]}
    assert (dst / "a-1" / "course.json").read_text() == "already here"


def test_a_missing_source_is_not_an_error(tmp_path):
    assert migrate.copy_courses(tmp_path / "nope", tmp_path / "dst") == {"copied": [], "skipped": []}


def test_defaults_to_laras_own_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path / "lara-courses")
    src = tmp_path / "src"; make(src, "a-1")
    assert migrate.copy_courses(src)["copied"] == ["a-1"] and (tmp_path / "lara-courses" / "a-1").exists()
