# 冲突处理记录

已在当前分支完成冲突检查与清理，处理方式如下：

1. 检查未完成合并状态：
   - `git status --porcelain=v1 -b`
2. 检查冲突标记（Git merge marker）：
   - `rg "^(<{7}|={7}|>{7})" -n`
3. 检查索引中是否存在 `U`（unmerged）文件：
   - `git diff --name-only --diff-filter=U`

当前结果：
- 未发现未合并文件。
- 未发现冲突标记。

若你后续将此分支与其他分支合并时再次出现冲突，请优先以
`shear_wall_pzt_damage_imaging.py` 的以下能力为保留主线：
- `--sensor-csv` 可配置阵列输入；
- 工程化参数 `MaterialConfig` 与 `ExcitationConfig`；
- `build_dataset_for_test_with_positions(...)` 的可复现实验数据流程。
