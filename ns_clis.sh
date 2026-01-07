# viser
ns-viewer --load-config outputs/<experiment-name>/<method-name>/<run-name>/config.yml

# garfield with custom train_split_fraction
ns-train garfield --data /your/data/here --pipeline.datamanager.dataparser.train_split_fraction 0.9

# train_split_fraction과 다른 설정을 함께 변경
ns-train garfield --data ./datasets/office \
  --pipeline.datamanager.dataparser.train_split_fraction 0.9 \
  --pipeline.datamanager.train_num_rays_per_batch 8192

ns-train garfield-gauss --data /your/data/here --pipeline.garfield-ckpt outputs/your/data/garfield/.../config.yml

ns-export pointcloud --load-config outputs/hand_hand/garfield/2025-12-23_135447/config.yml --output-dir outputs/hand_hand/garfield/2025-12-23_135447/exports/pcd/ --num-points 1000000 --remove-outliers True --normal-method open3d --save-world-frame False 
ns-export poisson --load-config outputs/hand_hand/garfield/2025-12-23_135447/config.yml --output-dir outputs/hand_hand/garfield/2025-12-23_135447/exports/mesh/ --target-num-faces 50000 --num-pixels-per-side 2048 --num-points 1000000 --remove-outliers True --normal-method open3d 

ns-train garfield-gauss --data ./datasets/lerf/dozer_nerfgun_waldo \
--pipeline.garfield-ckpt outputs/dozer_nerfgun_waldo/garfield/2025-12-19_174136/config.yml


ns-export pointcloud --load-config outputs/office/garfield/2025-12-24_111027/config.yml --output-dir outputs/office/garfield/2025-12-24_111027/exports/pcd/ --num-points 1000000 --remove-outliers True --normal-method open3d --save-world-frame False 
ns-export poisson --load-config outputs/office/garfield/2025-12-24_111027/config.yml --output-dir outputs/office/garfield/2025-12-24_111027/exports/mesh/ --target-num-faces 50000 --num-pixels-per-side 2048 --num-points 1000000 --remove-outliers True --normal-method open3d 
