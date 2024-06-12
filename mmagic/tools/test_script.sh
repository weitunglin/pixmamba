# Test Scripts
# ```
# visualizer = dict(
#     type='ConcatImageVisualizer',
#     fn_key='img_path',
#     img_keys=['pred_img'],
#     bgr2rgb=True)


# custom_hooks = [
#     dict(type='BasicVisualizationHook', interval=1)]
# ```
# add above test scripts into config

export UIEB_BASE=/home/allen/workspace/UIE_Benckmark
export UIEB_PATH=${UIEB_BASE}/data/UIEB/All_Results
export MODEL_VER=final_2
export CONFIG_PATH=configs/pixmamba/${MODEL_VER}.py
export EXP_NAME=pixmamba_uieb_${MODEL_VER}
export CKPT_ITER=_34200
export CKPT_PATH=work_dirs/${EXP_NAME}/iter${CKPT_ITER}.pth

PORT=29504 WANDB_MODE=offline bash ./tools/dist_test.sh $CONFIG_PATH $CKPT_PATH 1 --work-dir work_dirs/${EXP_NAME}/test

export RUN_NAME=20240610_001355
echo "Start evalutaing T90"
rm -rf ${UIEB_PATH}/${EXP_NAME}/T90
mkdir -p ${UIEB_PATH}/${EXP_NAME}/T90
cp -r work_dirs/${EXP_NAME}/test/${RUN_NAME}/vis_data/vis_image/t90_*.png ${UIEB_PATH}/${EXP_NAME}/T90
cd ${UIEB_PATH}/${EXP_NAME}/T90 && for f in *.png ; do f1=${f/$CKPT_ITER/} ; mv -- "$f" "${f1/t90_/}" ; done
cd ${UIEB_BASE} && python evaluate_UIEB.py --method_name ${EXP_NAME} --folder T90
cd /home/allen/workspace/seamamba/mmagic

echo "Start evalutaing C60"
rm -rf ${UIEB_PATH}/${EXP_NAME}/C60
mkdir -p ${UIEB_PATH}/${EXP_NAME}/C60
cp -r work_dirs/${EXP_NAME}/test/${RUN_NAME}/vis_data/vis_image/c60_*.png ${UIEB_PATH}/${EXP_NAME}/C60
cd ${UIEB_PATH}/${EXP_NAME}/C60 && for f in *.png ; do mv -- "$f" "${f/$CKPT_ITER/}" ; done
cd ${UIEB_BASE} && python evaluate_UIEB.py --method_name ${EXP_NAME} --folder C60
cd /home/allen/workspace/seamamba/mmagic

echo "Start evalutaing UCCS"
rm -rf ${UIEB_PATH}/${EXP_NAME}/UCCS
mkdir -p ${UIEB_PATH}/${EXP_NAME}/UCCS
cp -r work_dirs/${EXP_NAME}/test/${RUN_NAME}/vis_data/vis_image/green_*.png ${UIEB_PATH}/${EXP_NAME}/UCCS
cp -r work_dirs/${EXP_NAME}/test/${RUN_NAME}/vis_data/vis_image/blue_*.png ${UIEB_PATH}/${EXP_NAME}/UCCS
cp -r work_dirs/${EXP_NAME}/test/${RUN_NAME}/vis_data/vis_image/bg_*.png ${UIEB_PATH}/${EXP_NAME}/UCCS
cd ${UIEB_BASE} && python evaluate_UIEB.py --method_name ${EXP_NAME} --folder UCCS
cd /home/allen/workspace/seamamba/mmagic
