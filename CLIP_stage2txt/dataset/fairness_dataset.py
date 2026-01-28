"""
Fairness 데이터셋용 데이터로더
/workspace/datasets/fairness 하위에 있는 데이터셋들(celebdf, dfd, dfdc, ff++)을 위한 데이터로더

Stage 2: Deepfake Detection 학습을 위한 데이터셋
- Train: ff++ (FaceForensics++)
- Test: celebdf, dfd, dfdc (Cross-dataset evaluation)
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


class FairnessDataset(data.Dataset):
    """
    Fairness 데이터셋을 위한 데이터로더 클래스

    CSV 파일에서 직접 이미지 경로와 레이블을 읽어옵니다.
    """

    def __init__(self, config=None, mode='train', dataset_name=None):
        """
        Args:
            config (dict): 설정 딕셔너리
            mode (str): 'train', 'validation' 또는 'test'
            dataset_name (str): 데이터셋 이름 ('celebdf', 'dfd', 'dfdc', 'ff++')
        """
        self.config = config
        self.mode = mode
        self.dataset_name = dataset_name

        # Fairness 데이터셋 루트 경로
        self.fairness_root = config.get('fairness_root', '/workspace/datasets/fairness')

        # 데이터 리스트
        self.image_list = []
        self.label_list = []
        self.intersec_label_list = []
        self.video_name_list = []

        # Subgroup 정보 리스트 (성별 + 인종)
        self.gender_list = []  # 1: Male, -1: Female
        self.race_list = []    # 0: Asian, 1: Black, 2: White, 3: Other
        self.subgroup_list = []

        # CSV 파일에서 데이터 수집
        if mode == 'train':
            dataset_list = config.get('train_dataset', ['ff++'])
            if isinstance(dataset_list, str):
                dataset_list = [dataset_list]
            for ds_name in dataset_list:
                self._load_dataset_from_csv(ds_name)

        elif mode == 'validation':
            validation_dataset = config.get('validation_dataset', [])
            if isinstance(validation_dataset, list) and len(validation_dataset) > 0:
                ds_name = validation_dataset[0]
            else:
                ds_name = validation_dataset if isinstance(validation_dataset, str) else 'ff++'
            self._load_dataset_from_csv(ds_name)

        elif mode == 'test':
            test_dataset = config.get('test_dataset', [])
            if isinstance(test_dataset, list) and len(test_dataset) > 0:
                ds_name = test_dataset[0]
            else:
                ds_name = test_dataset if isinstance(test_dataset, str) else 'celebdf'
            self._load_dataset_from_csv(ds_name)

        else:
            raise NotImplementedError('Only train, validation, and test modes are supported.')

        assert len(self.image_list) != 0 and len(self.label_list) != 0, \
            f"No data collected for {mode} mode!"

        # 균형 샘플링 적용 여부
        use_balance_sampling = self.config.get('use_balance_sampling', False)
        use_subgroup_balance = self.config.get('use_subgroup_balance', False)

        if mode == 'train':
            if use_subgroup_balance:
                # Subgroup 간 균형 샘플링 (Subgroup 내 Real/Fake 균형도 포함)
                self._balance_subgroups()
            elif use_balance_sampling:
                # Subgroup 내 Real/Fake 균형만 맞춤 (기존 방식)
                self._balance_dataset()

        # 데이터 딕셔너리 생성
        self.data_dict = {
            'image': self.image_list,
            'label': self.label_list,
            'intersec_label': self.intersec_label_list,
            'video_name': self.video_name_list,
            'gender': self.gender_list,
            'race': self.race_list,
            'subgroup': self.subgroup_list,
        }

        # Data augmentation 초기화
        self.transform = self._init_data_aug_method()

        print(f"[FairnessDataset] Loaded {len(self.image_list)} images for {mode} mode")
        print(f"  Real: {sum(1 for l in self.label_list if l == 0)}, "
              f"Fake: {sum(1 for l in self.label_list if l == 1)}")

    def _load_dataset_from_csv(self, dataset_name):
        """CSV 파일에서 데이터셋 로드"""
        dataset_path = os.path.join(self.fairness_root, dataset_name)

        if self.mode == 'train':
            if dataset_name == 'ff++':
                csv_files = [
                    os.path.join(dataset_path, 'faketrain.csv'),
                    os.path.join(dataset_path, 'realtrain.csv')
                ]
            else:
                csv_files = [os.path.join(dataset_path, 'test.csv')]

        elif self.mode == 'validation':
            if dataset_name == 'ff++':
                csv_files = [
                    os.path.join(dataset_path, 'fakeval.csv'),
                    os.path.join(dataset_path, 'realval.csv')
                ]
            else:
                csv_files = [os.path.join(dataset_path, 'test.csv')]

        elif self.mode == 'test':
            csv_files = [os.path.join(dataset_path, 'test.csv')]

        else:
            raise Exception('Only train, validation, and test modes are supported.')

        for csv_file in csv_files:
            if not os.path.exists(csv_file):
                print(f"Warning: CSV file not found: {csv_file}")
                continue
            self._read_csv_file(csv_file, dataset_name)

    def _read_csv_file(self, csv_file, dataset_name):
        """CSV 파일을 읽어서 데이터 리스트에 추가"""
        df = pd.read_csv(csv_file)

        for idx, row in df.iterrows():
            img_path = row['img_path']
            label = int(row['label'])

            # intersec_label 컬럼 읽기
            intersec_label = int(row['intersec_label']) if 'intersec_label' in df.columns else -1

            # 성별 정보 읽기
            gender = int(row['ismale']) if 'ismale' in df.columns else 0

            # 인종 정보 읽기
            asian = int(row['isasian']) if 'isasian' in df.columns else 0
            white = int(row['iswhite']) if 'iswhite' in df.columns else 0
            black = int(row['isblack']) if 'isblack' in df.columns else 0

            # 인종 분류
            race = self._determine_race(asian, white, black)

            # Subgroup 분류
            subgroup = self._determine_subgroup(gender, race)

            # 성별이나 인종이 Unknown인 경우 스킵 (선택적)
            skip_unknown = self.config.get('skip_unknown_attributes', False)
            if skip_unknown and (gender == 0 or race == -1):
                continue

            # 절대 경로 생성
            full_img_path = os.path.join(self.fairness_root, img_path)

            # 비디오 이름 추출
            filename = os.path.basename(img_path)
            parts = filename.split('_')
            video_name = '_'.join(parts[1:-1]) if len(parts) >= 2 else filename

            self.image_list.append(full_img_path)
            self.label_list.append(label)
            self.intersec_label_list.append(intersec_label)
            self.video_name_list.append(video_name)
            self.gender_list.append(gender)
            self.race_list.append(race)
            self.subgroup_list.append(subgroup)

    def _determine_race(self, asian, white, black):
        """인종 분류"""
        if asian == 1:
            return 0  # Asian
        elif black == 1:
            return 1  # Black
        elif white == 1:
            return 2  # White
        elif asian == -1 and white == -1 and black == -1:
            return 3  # Other
        else:
            return -1  # Unknown

    def _determine_subgroup(self, gender, race):
        """Subgroup 분류"""
        if gender == 1:  # Male
            return race if race >= 0 else -1
        elif gender == -1:  # Female
            return race + 4 if race >= 0 else -1
        else:
            return -1

    def _balance_dataset(self):
        """Subgroup별 균형잡힌 데이터셋 생성 (Subgroup 내 Real/Fake 균형만 맞춤)"""
        print("\n[Subgroup Balanced Sampling - Within Subgroup]")
        selected_indices = []

        for subgroup_id in range(8):
            subgroup_indices = [i for i, sg in enumerate(self.subgroup_list) if sg == subgroup_id]
            if len(subgroup_indices) == 0:
                continue

            real_indices = [i for i in subgroup_indices if self.label_list[i] == 0]
            fake_indices = [i for i in subgroup_indices if self.label_list[i] == 1]

            if len(real_indices) == 0 or len(fake_indices) == 0:
                continue

            min_samples = min(len(real_indices), len(fake_indices))
            random.shuffle(real_indices)
            random.shuffle(fake_indices)

            selected_indices.extend(real_indices[:min_samples] + fake_indices[:min_samples])

        if len(selected_indices) > 0:
            random.shuffle(selected_indices)
            self._rebuild_dataset(selected_indices)

    def _balance_subgroups(self):
        """
        Subgroup 간 균형 샘플링
        모든 subgroup이 동일한 샘플 수를 갖도록 조정
        """
        print("\n[Subgroup Balanced Sampling - Between Subgroups]")

        # Subgroup 이름 정의
        subgroup_names = [
            "(Male, Asian)", "(Male, Black)", "(Male, White)", "(Male, Other)",
            "(Female, Asian)", "(Female, Black)", "(Female, White)", "(Female, Other)"
        ]

        # 1. 각 subgroup별 유효 샘플 수 계산 (Real/Fake 최소값 × 2)
        subgroup_counts = {}
        for sg_id in range(8):
            indices = [i for i, sg in enumerate(self.subgroup_list) if sg == sg_id]
            if len(indices) == 0:
                print(f"  Subgroup {sg_id} {subgroup_names[sg_id]}: No samples")
                continue

            real = sum(1 for i in indices if self.label_list[i] == 0)
            fake = sum(1 for i in indices if self.label_list[i] == 1)

            if real == 0 or fake == 0:
                print(f"  Subgroup {sg_id} {subgroup_names[sg_id]}: Real={real}, Fake={fake} (skipped - missing class)")
                continue

            subgroup_counts[sg_id] = min(real, fake) * 2  # Real:Fake 50:50 유지
            print(f"  Subgroup {sg_id} {subgroup_names[sg_id]}: Real={real}, Fake={fake}, Valid={subgroup_counts[sg_id]}")

        if len(subgroup_counts) == 0:
            print("  Warning: No valid subgroups found!")
            return

        # 2. 전체 subgroup 중 최소 샘플 수 결정
        min_samples = min(subgroup_counts.values())
        samples_per_class = min_samples // 2  # Real/Fake 각각

        print(f"\n  Target samples per subgroup: {min_samples} ({samples_per_class} Real + {samples_per_class} Fake)")

        # 3. 각 subgroup에서 동일 개수 샘플링
        selected_indices = []
        for sg_id in subgroup_counts.keys():
            sg_indices = [i for i, sg in enumerate(self.subgroup_list) if sg == sg_id]
            real_idx = [i for i in sg_indices if self.label_list[i] == 0]
            fake_idx = [i for i in sg_indices if self.label_list[i] == 1]

            random.shuffle(real_idx)
            random.shuffle(fake_idx)

            selected_indices.extend(real_idx[:samples_per_class])
            selected_indices.extend(fake_idx[:samples_per_class])

        # 4. 데이터셋 재구성
        if len(selected_indices) > 0:
            random.shuffle(selected_indices)
            self._rebuild_dataset(selected_indices)

        # 최종 분포 출력
        self._print_subgroup_distribution()

    def _rebuild_dataset(self, selected_indices):
        """선택된 인덱스로 데이터셋 재구성"""
        self.image_list = [self.image_list[i] for i in selected_indices]
        self.label_list = [self.label_list[i] for i in selected_indices]
        self.intersec_label_list = [self.intersec_label_list[i] for i in selected_indices]
        self.video_name_list = [self.video_name_list[i] for i in selected_indices]
        self.gender_list = [self.gender_list[i] for i in selected_indices]
        self.race_list = [self.race_list[i] for i in selected_indices]
        self.subgroup_list = [self.subgroup_list[i] for i in selected_indices]

    def _print_subgroup_distribution(self):
        """Subgroup 분포 출력"""
        subgroup_names = [
            "(Male, Asian)", "(Male, Black)", "(Male, White)", "(Male, Other)",
            "(Female, Asian)", "(Female, Black)", "(Female, White)", "(Female, Other)"
        ]

        print(f"\n[Final Subgroup Distribution] Total: {len(self.subgroup_list)} samples")

        for subgroup_id in range(8):
            indices = [i for i, sg in enumerate(self.subgroup_list) if sg == subgroup_id]
            real_count = sum(1 for i in indices if self.label_list[i] == 0)
            fake_count = sum(1 for i in indices if self.label_list[i] == 1)
            total_count = len(indices)

            if total_count > 0:
                print(f"  Subgroup {subgroup_id} {subgroup_names[subgroup_id]}: "
                      f"Total={total_count}, Real={real_count}, Fake={fake_count}")

    def _init_data_aug_method(self):
        """데이터 증강 초기화"""
        size = self.config.get('resolution', 256)

        if not self.config.get('use_data_augmentation', True) or self.mode in ['test', 'validation']:
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
        """이미지 정규화"""
        mean = self.config.get('mean', [0.485, 0.456, 0.406])
        std = self.config.get('std', [0.229, 0.224, 0.225])
        normalize = T.Normalize(mean=mean, std=std)
        return normalize(img)

    def __getitem__(self, index):
        """데이터 포인트 반환"""
        image_path = self.data_dict['image'][index]
        label = self.data_dict['label'][index]
        intersec_label = self.data_dict['intersec_label'][index]
        video_name = self.data_dict['video_name'][index]

        # 이미지 로드 (재시도 메커니즘)
        max_retries = 3
        for retry in range(max_retries):
            try:
                image = self.load_rgb(image_path)
                break
            except Exception as e:
                if retry == max_retries - 1:
                    fallback_index = (index + 1) % len(self.image_list)
                    return self.__getitem__(fallback_index)

        # 데이터 증강 적용
        transformed = self.transform(image=image)
        image = transformed['image']

        # 텐서로 변환 및 정규화
        image = self.normalize(self.to_tensor(image))

        return (image, label, None, None, None, None, None, None, video_name)

    @staticmethod
    def collate_fn(batch):
        """배치 데이터 collate"""
        (images, labels, landmarks, masks, xrays, patch_labels,
         clip_patch_labels, if_boundaries, video_names) = zip(*batch)

        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)

        data_dict = {
            'image': images,
            'label': labels,
            'landmark': None,
            'mask': None,
            'xray': None,
            'patch_label': None,
            'clip_patch_label': None,
            'if_boundary': None,
            'video_id': list(video_names),
        }

        return data_dict

    def __len__(self):
        return len(self.image_list)

    def get_subgroup_list(self):
        """Subgroup 리스트 반환"""
        return self.subgroup_list


class SubgroupBalancedBatchSampler(data.Sampler):
    """Subgroup별로 균등하게 샘플링하는 배치 샘플러"""

    def __init__(self, subgroup_list, batch_size, drop_last=False):
        self.subgroup_list = subgroup_list
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.subgroup_indices = {}
        for subgroup_id in range(8):
            self.subgroup_indices[subgroup_id] = [
                i for i, sg in enumerate(subgroup_list) if sg == subgroup_id
            ]

        self.valid_subgroups = [sg for sg in range(8) if len(self.subgroup_indices[sg]) > 0]
        self.num_valid_subgroups = len(self.valid_subgroups)

    def __iter__(self):
        samples_per_subgroup = max(1, self.batch_size // self.num_valid_subgroups)

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
        samples_per_subgroup = max(1, self.batch_size // self.num_valid_subgroups)
        min_size = min(len(self.subgroup_indices[sg]) for sg in self.valid_subgroups)
        return min_size // samples_per_subgroup
