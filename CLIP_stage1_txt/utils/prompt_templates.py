"""
Prompt Templates for CLIP Stage1 Text-based Fairness Learning

Stage1용 Subgroup 프롬프트 정의 (Real/Fake 구분 없이 demographic 기반)

성별 인코딩 (CLIP_stage1 방식):
- 0: Male
- 1: Female

인종 인코딩:
- 0: Asian
- 1: Black
- 2: White
- 3: Other

Subgroup 매핑 (gender * 4 + race):
- 0: Male, Asian
- 1: Male, Black
- 2: Male, White
- 3: Male, Other
- 4: Female, Asian
- 5: Female, Black
- 6: Female, White
- 7: Female, Other
"""

from typing import List, Dict


# ========== Subgroup별 프롬프트 (demographic 기반) ==========
# 각 subgroup에 대해 6개의 다양한 프롬프트 제공

SUBGROUP_PROMPTS: Dict[int, List[str]] = {
    # Male, Asian (subgroup 0)
    0: [
        "A photo of an Asian man",
        "An Asian male face",
        "Portrait of an Asian male",
        "Face of a man of Asian descent",
        "An Asian man's face",
        "A male person with Asian features",
    ],
    # Male, Black (subgroup 1)
    1: [
        "A photo of a Black man",
        "A Black male face",
        "Portrait of a Black male",
        "Face of a man of African descent",
        "A Black man's face",
        "A male person with African features",
    ],
    # Male, White (subgroup 2)
    2: [
        "A photo of a White man",
        "A White male face",
        "Portrait of a Caucasian male",
        "Face of a man of European descent",
        "A Caucasian man's face",
        "A male person with Caucasian features",
    ],
    # Male, Other (subgroup 3)
    3: [
        "A photo of a man",
        "A male face",
        "Portrait of a male",
        "Face of a man",
        "A man's face",
        "A male person",
    ],
    # Female, Asian (subgroup 4)
    4: [
        "A photo of an Asian woman",
        "An Asian female face",
        "Portrait of an Asian female",
        "Face of a woman of Asian descent",
        "An Asian woman's face",
        "A female person with Asian features",
    ],
    # Female, Black (subgroup 5)
    5: [
        "A photo of a Black woman",
        "A Black female face",
        "Portrait of a Black female",
        "Face of a woman of African descent",
        "A Black woman's face",
        "A female person with African features",
    ],
    # Female, White (subgroup 6)
    6: [
        "A photo of a White woman",
        "A White female face",
        "Portrait of a Caucasian female",
        "Face of a woman of European descent",
        "A Caucasian woman's face",
        "A female person with Caucasian features",
    ],
    # Female, Other (subgroup 7)
    7: [
        "A photo of a woman",
        "A female face",
        "Portrait of a female",
        "Face of a woman",
        "A woman's face",
        "A female person",
    ],
}


# ========== Subgroup 매핑 정보 ==========
GENDER_MAPPING = {
    0: 'male',
    1: 'female',
}

RACE_MAPPING = {
    0: 'asian',
    1: 'black',
    2: 'white',
    3: 'other',
}

SUBGROUP_MAPPING = {
    0: ('male', 'asian'),
    1: ('male', 'black'),
    2: ('male', 'white'),
    3: ('male', 'other'),
    4: ('female', 'asian'),
    5: ('female', 'black'),
    6: ('female', 'white'),
    7: ('female', 'other'),
}

SUBGROUP_NAMES = {
    0: 'Male Asian',
    1: 'Male Black',
    2: 'Male White',
    3: 'Male Other',
    4: 'Female Asian',
    5: 'Female Black',
    6: 'Female White',
    7: 'Female Other',
}


def get_subgroup_prompts(num_prompts_per_subgroup: int = 6) -> Dict[int, List[str]]:
    """
    각 subgroup에 대한 프롬프트 반환

    Args:
        num_prompts_per_subgroup: 각 subgroup당 반환할 프롬프트 수 (최대 6)

    Returns:
        Dict[int, List[str]]: subgroup_id -> prompts 매핑
    """
    result = {}
    for subgroup_id, prompts in SUBGROUP_PROMPTS.items():
        result[subgroup_id] = prompts[:num_prompts_per_subgroup]
    return result


def get_prompts_for_subgroup(subgroup_id: int) -> List[str]:
    """
    특정 subgroup의 모든 프롬프트 반환

    Args:
        subgroup_id: 0-7 범위의 subgroup ID

    Returns:
        List[str]: 해당 subgroup의 프롬프트 리스트
    """
    return SUBGROUP_PROMPTS.get(subgroup_id, SUBGROUP_PROMPTS[3])  # fallback to Male Other


def compute_subgroup_id(gender: int, race: int) -> int:
    """
    성별과 인종으로부터 subgroup ID 계산

    Args:
        gender: 0 (Male) or 1 (Female)
        race: 0 (Asian), 1 (Black), 2 (White), 3 (Other)

    Returns:
        int: subgroup ID (0-7)
    """
    return gender * 4 + race


def get_subgroup_name(subgroup_id: int) -> str:
    """
    Subgroup ID에 대한 이름 반환

    Args:
        subgroup_id: 0-7 범위의 subgroup ID

    Returns:
        str: subgroup 이름 (예: "Male Asian")
    """
    return SUBGROUP_NAMES.get(subgroup_id, "Unknown")


# ========== 테스트 ==========
if __name__ == "__main__":
    print("=" * 80)
    print("CLIP Stage1 Text-based Fairness - Prompt Templates")
    print("=" * 80)

    # 각 subgroup의 프롬프트 출력
    for subgroup_id in range(8):
        prompts = get_prompts_for_subgroup(subgroup_id)
        name = get_subgroup_name(subgroup_id)
        print(f"\n[Subgroup {subgroup_id}] {name}:")
        for i, prompt in enumerate(prompts, 1):
            print(f"  {i}. {prompt}")

    # Subgroup ID 계산 테스트
    print("\n" + "=" * 80)
    print("Subgroup ID Calculation Test:")
    print("=" * 80)
    for gender in [0, 1]:
        for race in range(4):
            sg_id = compute_subgroup_id(gender, race)
            print(f"  gender={gender}, race={race} -> subgroup_id={sg_id} ({get_subgroup_name(sg_id)})")

    print("\n" + "=" * 80)
    print("Prompt templates created successfully!")
    print("=" * 80)
