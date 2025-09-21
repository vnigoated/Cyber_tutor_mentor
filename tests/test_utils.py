import sys
import os
# ensure project root is on sys.path for tests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from utils import parse_json_from_model, grade_quiz_struct


def test_parse_json_from_clean_string():
    text = '[{"question":"Q1","options":["o1","o2","o3","o4"],"answer":"A","explanation":"e"}]'
    parsed = parse_json_from_model(text)
    assert isinstance(parsed, list)
    assert parsed[0]['question'] == 'Q1'


def test_parse_json_from_fenced_string():
    text = '```json\n[{"question":"Q2","options":["o1","o2","o3","o4"],"answer":"B","explanation":"e2"}]\n```'
    parsed = parse_json_from_model(text)
    assert isinstance(parsed, list)
    assert parsed[0]['answer'] == 'B'


def test_grade_quiz_struct():
    quiz = [
        {'question':'Q1','options':['opt1','opt2','opt3','opt4'],'answer':'A','explanation':'ex1'},
        {'question':'Q2','options':['o1','o2','o3','o4'],'answer':'C','explanation':'ex2'}
    ]
    answers = ['A. opt1', 'B. o2']
    result = grade_quiz_struct(quiz, answers)
    assert result['score'] == 1
    assert result['total'] == 2
    assert result['details'][0]['is_correct'] is True
    assert result['details'][1]['is_correct'] is False
