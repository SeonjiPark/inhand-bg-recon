# # viser
ns-train garfield --data ./datasets/hand_hand --pipeline.train_split_fraction 0.9 --viewer.quit-on-train-completion True
ns-viewer --load-config outputs/<experiment-name>/<method-name>/<run-name>/config.yml

# ns-render camera-path --load-config outputs/office/garfield/2025-12-29_175148/config.yml --camera-path-filename /data/seonjipark/Research/garfield/datasets/office/camera_paths/2026-01-15-18-33-12.json --output-path renders/office/2026-01-15-18-33-12.mp4
# ns-export pointcloud --load-config outputs/office/garfield/2025-12-29_175148/config.yml --output-dir exports/pcd/office_table --num-points 1000000 --remove-outliers True --normal-method open3d --save-world-frame False 
# ns-export pointcloud --load-config outputs/office/garfield/2025-12-29_175148/config.yml --output-dir exports/pcd/office_table --num-points 1000000 --remove-outliers True --normal-method open3d --save-world-frame False 


# ns-train garfield --data ./datasets/lerf/dozer_nerfgun_waldo --pipeline.train_split_fraction 0.9
# ns-train garfield --data ./datasets/lerf/dozer_nerfgun_waldo --pipeline.train_split_fraction 0.5 && \
# ns-train garfield --data ./datasets/lerf/dozer_nerfgun_waldo --pipeline.train_split_fraction 0.1 --viewer.quit-on-train-complete true       
# ns-train garfield --data ./datasets/office0 --pipeline.train_split_fraction 0.9 --viewer.quit-on-train-complete true
# ns-train garfield --data ./datasets/office0 --pipeline.train_split_fraction 0.5 --pipeline.model.train_scheduling 2
# ns-train garfield --data ./datasets/office0 --pipeline.train_split_fraction 0.1 --pipeline.model.train_scheduling 2

# ns-train garfield --data ./datasets/hand_hand --pipeline.train_split_fraction 0.1 --pipeline.model.train_scheduling 2 --viewer.quit-on-train-completion True
# ns-train garfield --data ./datasets/hand_hand --pipeline.train_split_fraction 0.5 --pipeline.model.train_scheduling 2 --viewer.quit-on-train-completion True

# ns-train garfield --data ./datasets/hand_hand --pipeline.train_split_fraction 0.1 --pipeline.model.train_scheduling 3 --viewer.quit-on-train-completion True
# ns-train garfield --data ./datasets/hand_hand --pipeline.train_split_fraction 0.1 --pipeline.model.train_scheduling 4 --viewer.quit-on-train-completion True
# ns-train garfield --data ./datasets/hand_hand --pipeline.train_split_fraction 0.1 --pipeline.model.train_scheduling 5 --viewer.quit-on-train-completion True

# ns-eval --load-config ./outputs/hand_hand/garfield2d/2026-01-27_175739/config.yml --output-path ./outputs/hand_hand/garfield2d/2026-01-27_175739/eval_results.json
# ns-eval --load-config ./outputs/hand_hand/garfield2d/2026-01-27_185254/config.yml --output-path ./outputs/hand_hand/garfield2d/2026-01-27_185254/eval_results.json
# ns-eval --load-config ./outputs/hand_hand/garfield2d/2026-01-27_192904ß/config.yml --output-path ./outputs/hand_hand/garfield2d/2026-01-27_192904/eval_results.json

