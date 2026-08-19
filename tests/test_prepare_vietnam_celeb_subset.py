from training.prepare_vietnam_celeb_subset import speaker_from_train_line


def test_parses_official_vietnam_celeb_train_list_row():
    assert speaker_from_train_line("id00000\t00000.wav") == "id00000"


def test_parses_path_based_train_list_row():
    assert speaker_from_train_line("id00001/00001.wav") == "id00001"
