"""
Fairness Stage1 데이터셋
FairFace, UTKFace, CausalFace 데이터셋을 통합하여 CLIP pre-train weight의 Global bias 제거

Subgroup 정의:
- 인종 (4): Asian(0), White(1), Black(2), Other(3)
- 성별 (2): Male(0), Female(1)
- 총 8개 subgroup: (Male,Asian)=0, (Male,White)=1, ..., (Female,Other)=7

데이터셋 경로:
- FairFace: /workspace/datasets/fairface
- UTKFace: /workspace/datasets/UTKFace
- CausalFace: /workspace/datasets/CausalFace
"""

import os
import random
import cv2
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms as T
import albumentations as A

from dataset.albu import IsotropicResize, CustomRandomCrop


# Race 매핑 (PG-FDD 통일 형식: 0=Asian, 1=White, 2=Black, 3=Other)
FAIRFACE_RACE_MAP = {
    'East Asian': 0, 'Southeast Asian': 0,  # Asian
    'White': 1,                              # White
    'Black': 2,                              # Black
    'Indian': 3, 'Middle Eastern': 3, 'Latino_Hispanic': 3  # Other
}

UTKFACE_RACE_MAP = {
    2: 0,  # Asian -> 0
    0: 1,  # White -> 1
    1: 2,  # Black -> 2
    3: 3,  # Indian -> Other
    4: 3,  # Others -> Other
}

CAUSALFACE_RACE_MAP = {
    'asian': 0,
    'white': 1,
    'black': 2,
    'indian': 3,
}

# Gender 매핑 (통일된 형식: 0=Male, 1=Female)
FAIRFACE_GENDER_MAP = {'Male': 0, 'Female': 1}
UTKFACE_GENDER_MAP = {0: 0, 1: 1}  # 0=male, 1=female
CAUSALFACE_GENDER_MAP = {'male': 0, 'female': 1}

# FF++ 데이터셋용 매핑 (ismale: 1=Male, -1=Female)
FFPP_GENDER_MAP = {1: 0, -1: 1}  # 1->Male(0), -1->Female(1)


class FairnessDataset(data.Dataset):
    """
    Fairness Stage1 학습을 위한 통합 데이터셋

    FairFace, UTKFace, CausalFace 데이터셋을 통합하여
    CLIP pre-train weight의 Global bias 제거를 위한 학습 데이터 제공

    Subgroup 레이블:
        - gender: 0 (Male), 1 (Female)
        - race: 0 (Asian), 1 (White), 2 (Black), 3 (Other)
        - subgroup: 0-7 (gender * 4 + race)
    """

    def __init__(self, config, mode='train'):
        """
        Args:
            config (dict): 설정 딕셔너리
            mode (str): 'train', 'validation', 또는 'test'
        """
        self.config = config
        self.mode = mode

        # 데이터셋 경로
        self.fairface_root = config.get('fairface_root', '/workspace/datasets/fairface')
        self.utkface_root = config.get('utkface_root', '/workspace/datasets/UTKFace')
        self.causalface_root = config.get('causalface_root', '/workspace/datasets/CausalFace')
        self.ffpp_root = config.get('ffpp_root', '/workspace/datasets/fairness/ff++')

        # 데이터 리스트
        self.image_list = []
        self.gender_list = []       # 0: Male, 1: Female
        self.race_list = []         # 0: Asian, 1: White, 2: Black, 3: Other
        self.subgroup_list = []     # 0-7

        # 데이터 로드
        if mode == 'train':
            # Train: FairFace, UTKFace, CausalFace
            datasets_to_use = config.get('train_datasets', ['fairface', 'utkface', 'causalface'])
            if 'fairface' in datasets_to_use:
                self._load_fairface()
            if 'utkface' in datasets_to_use:
                self._load_utkface()
            if 'causalface' in datasets_to_use:
                self._load_causalface()
        else:
            # Validation/Test: FF++ 데이터셋 사용
            use_ffpp_for_val = config.get('use_ffpp_for_validation', True)
            if use_ffpp_for_val:
                self._load_ffpp()
            else:
                # 기존 방식 (FairFace, UTKFace, CausalFace의 validation split)
                datasets_to_use = config.get('train_datasets', ['fairface', 'utkface', 'causalface'])
                if 'fairface' in datasets_to_use:
                    self._load_fairface()
                if 'utkface' in datasets_to_use:
                    self._load_utkface()
                if 'causalface' in datasets_to_use:
                    self._load_causalface()

        # 데이터 검증
        assert len(self.image_list) > 0, f"No data collected for {mode} mode!"

        # 균형 샘플링 적용 (train 모드만)
        use_balance_sampling = config.get('use_balance_sampling', True)
        if use_balance_sampling and mode == 'train':
            self._balance_dataset()

        # 데이터 딕셔너리 생성
        self.data_dict = {
            'image': self.image_list,
            'gender': self.gender_list,
            'race': self.race_list,
            'subgroup': self.subgroup_list,
        }

        # 데이터 증강 초기화
        self.transform = self._init_data_aug_method()

        # 통계 출력
        self._print_statistics()

    def _load_fairface(self):
        """FairFace 데이터셋 로드"""
        print(f"[FairnessDataset] Loading FairFace ({self.mode})...")

        if self.mode == 'train':
            csv_path = os.path.join(self.fairface_root, 'fairface_label_train.csv')
        else:  # validation or test
            csv_path = os.path.join(self.fairface_root, 'fairface_label_val.csv')

        if not os.path.exists(csv_path):
            print(f"  Warning: {csv_path} not found, skipping FairFace")
            return

        df = pd.read_csv(csv_path)
        count = 0

        for _, row in df.iterrows():
            file_path = row['file']
            gender_str = row['gender']
            race_str = row['race']

            # 매핑
            gender = FAIRFACE_GENDER_MAP.get(gender_str)
            race = FAIRFACE_RACE_MAP.get(race_str)

            if gender is None or race is None:
                continue

            # 절대 경로 생성
            full_path = os.path.join(self.fairface_root, file_path)
            if not os.path.exists(full_path):
                continue

            # Subgroup 계산
            subgroup = gender * 4 + race

            self.image_list.append(full_path)
            self.gender_list.append(gender)
            self.race_list.append(race)
            self.subgroup_list.append(subgroup)
            count += 1

        print(f"  Loaded {count} images from FairFace")

    def _load_utkface(self):
        """UTKFace 데이터셋 로드"""
        print(f"[FairnessDataset] Loading UTKFace ({self.mode})...")

        # UTKFace는 part1, part2, part3으로 나뉨
        parts = ['part1', 'part2', 'part3']

        # train/val/test 분할 (8:1:1)
        all_images = []
        for part in parts:
            part_path = os.path.join(self.utkface_root, part)
            if os.path.exists(part_path):
                images = [os.path.join(part_path, f) for f in os.listdir(part_path)
                         if f.endswith('.jpg') or f.endswith('.png')]
                all_images.extend(images)

        # 정렬 후 분할 (일관성 유지)
        all_images.sort()
        n = len(all_images)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        if self.mode == 'train':
            selected_images = all_images[:train_end]
        elif self.mode == 'validation':
            selected_images = all_images[train_end:val_end]
        else:  # test
            selected_images = all_images[val_end:]

        count = 0
        for img_path in selected_images:
            # 파일명에서 레이블 추출: [age]_[gender]_[race]_[date].jpg
            filename = os.path.basename(img_path)
            parts = filename.split('_')

            if len(parts) < 3:
                continue

            try:
                gender_raw = int(parts[1])
                race_raw = int(parts[2])
            except (ValueError, IndexError):
                continue

            # 매핑
            gender = UTKFACE_GENDER_MAP.get(gender_raw)
            race = UTKFACE_RACE_MAP.get(race_raw)

            if gender is None or race is None:
                continue

            # Subgroup 계산
            subgroup = gender * 4 + race

            self.image_list.append(img_path)
            self.gender_list.append(gender)
            self.race_list.append(race)
            self.subgroup_list.append(subgroup)
            count += 1

        print(f"  Loaded {count} images from UTKFace")

    def _load_causalface(self):
        """CausalFace 데이터셋 로드"""
        print(f"[FairnessDataset] Loading CausalFace ({self.mode})...")

        # final_picked_* 폴더들에서 이미지 수집
        folders = ['final_picked_age', 'final_picked_lighting',
                   'final_picked_pose', 'final_picked_smiling']

        all_images = []
        for folder in folders:
            folder_path = os.path.join(self.causalface_root, folder)
            if not os.path.exists(folder_path):
                continue

            # seed_* 서브폴더들
            for seed_dir in os.listdir(folder_path):
                seed_path = os.path.join(folder_path, seed_dir)
                if not os.path.isdir(seed_path):
                    continue

                for img_file in os.listdir(seed_path):
                    if img_file.endswith('.png') and '_rm_bg' not in img_file:
                        img_path = os.path.join(seed_path, img_file)
                        all_images.append((img_path, img_file))

        # 중복 제거 및 정렬
        all_images = list(set(all_images))
        all_images.sort(key=lambda x: x[0])

        # train/val/test 분할 (8:1:1)
        n = len(all_images)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)

        if self.mode == 'train':
            selected = all_images[:train_end]
        elif self.mode == 'validation':
            selected = all_images[train_end:val_end]
        else:  # test
            selected = all_images[val_end:]

        count = 0
        for img_path, img_file in selected:
            # 파일명에서 레이블 추출: race_gender_age_*.png
            # 예: asian_female_age_0.8_o2.png
            name_lower = img_file.lower()

            # Race 추출
            race = None
            for race_name, race_id in CAUSALFACE_RACE_MAP.items():
                if race_name in name_lower:
                    race = race_id
                    break

            # Gender 추출
            gender = None
            for gender_name, gender_id in CAUSALFACE_GENDER_MAP.items():
                if gender_name in name_lower:
                    gender = gender_id
                    break

            if gender is None or race is None:
                continue

            # Subgroup 계산
            subgroup = gender * 4 + race

            self.image_list.append(img_path)
            self.gender_list.append(gender)
            self.race_list.append(race)
            self.subgroup_list.append(subgroup)
            count += 1

        print(f"  Loaded {count} images from CausalFace")

    def _load_ffpp(self):
        """FF++ 데이터셋 로드 (Validation/Test용)"""
        print(f"[FairnessDataset] Loading FF++ ({self.mode})...")

        # CSV 파일 결정
        if self.mode == 'validation':
            csv_files = [
                os.path.join(self.ffpp_root, 'fakeval.csv'),
                os.path.join(self.ffpp_root, 'realval.csv')
            ]
        elif self.mode == 'test':
            csv_files = [os.path.join(self.ffpp_root, 'test.csv')]
        else:  # train (Stage2에서 사용할 수 있음)
            csv_files = [
                os.path.join(self.ffpp_root, 'faketrain.csv'),
                os.path.join(self.ffpp_root, 'realtrain.csv')
            ]

        count = 0
        for csv_path in csv_files:
            if not os.path.exists(csv_path):
                print(f"  Warning: {csv_path} not found")
                continue

            df = pd.read_csv(csv_path)

            for _, row in df.iterrows():
                img_path = row['img_path']

                # 성별 정보 (ismale: 1=Male, -1=Female, 0 or missing=Unknown)
                gender_raw = int(row.get('ismale', 0))
                if gender_raw == 0:
                    continue  # Unknown 제외
                gender = FFPP_GENDER_MAP.get(gender_raw)

                # 인종 정보
                asian = int(row.get('isasian', 0))
                white = int(row.get('iswhite', 0))
                black = int(row.get('isblack', 0))
                race = self._determine_race_ffpp(asian, white, black)

                if gender is None or race is None or race == -1:
                    continue

                # Subgroup 계산
                subgroup = gender * 4 + race

                # 절대 경로 생성 (ff++/crop_img/... -> /workspace/datasets/fairness/ff++/crop_img/...)
                if img_path.startswith('ff++/'):
                    full_path = os.path.join('/workspace/datasets/fairness', img_path)
                else:
                    full_path = os.path.join(self.ffpp_root, img_path)

                if not os.path.exists(full_path):
                    continue

                self.image_list.append(full_path)
                self.gender_list.append(gender)
                self.race_list.append(race)
                self.subgroup_list.append(subgroup)
                count += 1

        print(f"  Loaded {count} images from FF++")

    def _determine_race_ffpp(self, asian, white, black):
        """FF++ 데이터셋용 인종 결정 (PG-FDD 기준: Asian=0, White=1, Black=2, Other=3)"""
        if asian == 1:
            return 0  # Asian
        elif white == 1:
            return 1  # White
        elif black == 1:
            return 2  # Black
        elif asian == -1 and white == -1 and black == -1:
            return 3  # Other
        else:
            return -1  # Unknown

    def _balance_dataset(self):
        """Subgroup별 균형잡힌 데이터셋 생성"""
        print("\n[Subgroup Balanced Sampling]")

        subgroup_names = [
            "(Male, Asian)", "(Male, White)", "(Male, Black)", "(Male, Other)",
            "(Female, Asian)", "(Female, White)", "(Female, Black)", "(Female, Other)"
        ]

        selected_indices = []

        # 각 subgroup의 최소 샘플 수 찾기
        subgroup_counts = {}
        for sg_id in range(8):
            subgroup_indices = [i for i, sg in enumerate(self.subgroup_list) if sg == sg_id]
            subgroup_counts[sg_id] = len(subgroup_indices)
            print(f"  Subgroup {sg_id} {subgroup_names[sg_id]}: {len(subgroup_indices)} samples")

        # 가장 적은 subgroup 기준으로 샘플링 (or 최대 샘플 수 제한)
        min_samples = min(subgroup_counts.values()) if min(subgroup_counts.values()) > 0 else 0
        max_samples_per_subgroup = self.config.get('max_samples_per_subgroup', min_samples)
        target_samples = min(min_samples, max_samples_per_subgroup) if max_samples_per_subgroup > 0 else min_samples

        print(f"  Target samples per subgroup: {target_samples}")

        for sg_id in range(8):
            subgroup_indices = [i for i, sg in enumerate(self.subgroup_list) if sg == sg_id]

            if len(subgroup_indices) == 0:
                continue

            random.shuffle(subgroup_indices)
            selected = subgroup_indices[:target_samples]
            selected_indices.extend(selected)

        if len(selected_indices) == 0:
            print("  Warning: No samples selected after balancing!")
            return

        random.shuffle(selected_indices)

        # 리스트 재구성
        self.image_list = [self.image_list[i] for i in selected_indices]
        self.gender_list = [self.gender_list[i] for i in selected_indices]
        self.race_list = [self.race_list[i] for i in selected_indices]
        self.subgroup_list = [self.subgroup_list[i] for i in selected_indices]

    def _init_data_aug_method(self):
        """데이터 증강 초기화"""
        size = self.config.get('resolution', 224)

        if self.mode != 'train' or not self.config.get('use_data_augmentation', True):
            trans = A.Compose([
                A.Resize(height=size, width=size, interpolation=cv2.INTER_LINEAR)
            ])
        else:
            trans = A.Compose([
                A.ImageCompression(quality_range=(40, 100), p=0.1),
                A.HorizontalFlip(p=0.5),
                A.GaussNoise(p=0.2),
                A.OneOf([
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA,
                                   interpolation_up=cv2.INTER_CUBIC),
                    IsotropicResize(max_side=size, interpolation_down=cv2.INTER_AREA,
                                   interpolation_up=cv2.INTER_LINEAR),
                    CustomRandomCrop(size=size)
                ], p=1),
                A.Resize(height=size, width=size),
                A.OneOf([
                    A.RandomBrightnessContrast(),
                    A.HueSaturationValue()
                ], p=0.3),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10,
                                   border_mode=cv2.BORDER_CONSTANT, p=0.3),
            ])
        return trans

    def _print_statistics(self):
        """데이터셋 통계 출력"""
        print(f"\n[FairnessDataset] Mode: {self.mode}, Total: {len(self.image_list)} images")

        subgroup_names = [
            "(Male, Asian)", "(Male, White)", "(Male, Black)", "(Male, Other)",
            "(Female, Asian)", "(Female, White)", "(Female, Black)", "(Female, Other)"
        ]

        for sg_id in range(8):
            count = sum(1 for sg in self.subgroup_list if sg == sg_id)
            if count > 0:
                print(f"  Subgroup {sg_id} {subgroup_names[sg_id]}: {count}")

    def load_rgb(self, file_path):
        """RGB 이미지 로드"""
        assert os.path.exists(file_path), f"{file_path} does not exist"

        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f'Loaded image is None: {file_path}')

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def to_tensor(self, img):
        """이미지를 PyTorch 텐서로 변환"""
        return T.ToTensor()(img)

    def normalize(self, img):
        """이미지 정규화 (CLIP 기본 정규화 사용)"""
        mean = self.config.get('mean', [0.48145466, 0.4578275, 0.40821073])
        std = self.config.get('std', [0.26862954, 0.26130258, 0.27577711])
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def __getitem__(self, index):
        """데이터 포인트 반환"""
        image_path = self.data_dict['image'][index]
        gender = self.data_dict['gender'][index]
        race = self.data_dict['race'][index]
        subgroup = self.data_dict['subgroup'][index]

        # 이미지 로드 (재시도 메커니즘)
        max_retries = 3
        for retry in range(max_retries):
            try:
                image = self.load_rgb(image_path)
                break
            except Exception:
                if retry == max_retries - 1:
                    fallback_index = (index + 1) % len(self.image_list)
                    return self.__getitem__(fallback_index)

        # 데이터 증강 적용
        transformed = self.transform(image=image)
        image = transformed['image']

        # 텐서로 변환 및 정규화
        image = self.normalize(self.to_tensor(image))

        return {
            'image': image,
            'gender': gender,
            'race': race,
            'subgroup': subgroup,
            'img_path': image_path
        }

    @staticmethod
    def collate_fn(batch):
        """배치 데이터 collate"""
        images = torch.stack([item['image'] for item in batch], dim=0)
        genders = torch.LongTensor([item['gender'] for item in batch])
        races = torch.LongTensor([item['race'] for item in batch])
        subgroups = torch.LongTensor([item['subgroup'] for item in batch])
        img_paths = [item['img_path'] for item in batch]

        return {
            'image': images,
            'gender': genders,
            'race': races,
            'subgroup': subgroups,
            'img_path': img_paths
        }

    def __len__(self):
        return len(self.image_list)

    def get_subgroup_list(self):
        """Subgroup 리스트 반환 (Sampler에서 사용)"""
        return self.subgroup_list


class SubgroupBalancedBatchSampler(data.Sampler):
    """
    Subgroup별로 균등하게 샘플링하는 배치 샘플러

    각 배치에 8개 subgroup이 균등하게 포함되도록 샘플링합니다.
    배치 크기는 8의 배수여야 합니다.
    """

    def __init__(self, subgroup_list, batch_size, drop_last=False):
        """
        Args:
            subgroup_list (list): 각 샘플의 subgroup ID (0-7)
            batch_size (int): 배치 크기 (8의 배수 권장)
            drop_last (bool): 마지막 불완전한 배치를 버릴지 여부
        """
        self.subgroup_list = subgroup_list
        self.batch_size = batch_size
        self.drop_last = drop_last

        # 각 subgroup별 인덱스 추출
        self.subgroup_indices = {}
        for subgroup_id in range(8):
            self.subgroup_indices[subgroup_id] = [
                i for i, sg in enumerate(subgroup_list) if sg == subgroup_id
            ]

        # 유효한 subgroup 찾기 (샘플이 있는 subgroup만)
        self.valid_subgroups = [sg for sg in range(8) if len(self.subgroup_indices[sg]) > 0]
        self.num_valid_subgroups = len(self.valid_subgroups)

        print(f"\n[SubgroupBalancedBatchSampler]")
        print(f"  Batch size: {batch_size}")
        print(f"  Valid subgroups: {self.num_valid_subgroups}")

    def __iter__(self):
        """배치 생성 이터레이터"""
        samples_per_subgroup = max(1, self.batch_size // self.num_valid_subgroups)

        # 각 subgroup의 인덱스를 섞기
        shuffled_indices = {}
        for sg_id in self.valid_subgroups:
            indices = self.subgroup_indices[sg_id].copy()
            random.shuffle(indices)
            shuffled_indices[sg_id] = indices

        subgroup_pointers = {sg_id: 0 for sg_id in self.valid_subgroups}

        min_size = min(len(shuffled_indices[sg]) for sg in self.valid_subgroups)
        num_batches = min_size // samples_per_subgroup

        for _ in range(num_batches):
            batch = []
            for sg_id in self.valid_subgroups:
                start = subgroup_pointers[sg_id]
                end = start + samples_per_subgroup

                if end > len(shuffled_indices[sg_id]):
                    indices = self.subgroup_indices[sg_id].copy()
                    random.shuffle(indices)
                    shuffled_indices[sg_id] = indices
                    subgroup_pointers[sg_id] = 0
                    start = 0
                    end = samples_per_subgroup

                batch.extend(shuffled_indices[sg_id][start:end])
                subgroup_pointers[sg_id] = end

            random.shuffle(batch)
            yield batch

    def __len__(self):
        """총 배치 수 계산"""
        samples_per_subgroup = max(1, self.batch_size // self.num_valid_subgroups)
        min_size = min(len(self.subgroup_indices[sg]) for sg in self.valid_subgroups)
        return min_size // samples_per_subgroup
