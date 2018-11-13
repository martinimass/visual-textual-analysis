net=$1

path="../../caffe/models/"
solver=""
model=""
weights=""
path_log="../results/visual/"
log=""

case $net in

"VGG16" )
	model="${path}VGG16/"
	solver="${model}retail/solver.prototxt"
	weights="${model}VGG_ILSVRC_16_layers.caffemodel"
	log="${path_log}finetuning-VGG16.log"
	;;

"AlexNet" )	
	model="${path}bvlc_alexnet/"
	solver="${model}retail/solver.prototxt"
	weights="${model}bvlc_alexnet.caffemodel"
	log="${path_log}finetuning-AlexNet.log"
	;;

"CaffeNet" )	
	model="${path}bvlc_reference_caffenet/"
	solver="${model}retail/solver.prototxt"
	weights="${model}bvlc_reference_caffenet.caffemodel"
	log="${path_log}finetuning-CaffeNet.log"
	;;

"GoogLeNet" )	
	model="${path}bvlc_googlenet/"
	solver="${model}retail/solver.prototxt"
	weights="${model}bvlc_googlenet.caffemodel"
	log="${path_log}finetuning-GoogLeNet.log"
	;;

"ResNet50" )
	model="${path}resnet50/"
	solver="${model}retail/solver.prototxt"
	weights="${model}resnet50.caffemodel"
	log="${path_log}finetuning-ResNet50.log"
	;;

"ResNet101" )
	model="${path}resnet101/"
	solver="${model}retail/solver.prototxt"
	weights="${model}resnet101.caffemodel"
	log="${path_log}finetuning-ResNet101.log"
	;;

"ResNet152" )
	model="${path}resnet152/"
	solver="${model}retail/solver.prototxt"
	weights="${model}resnet152.caffemodel"
	log="${path_log}finetuning-ResNet152.log"
	;;

*)	echo "Supported CNNs: VGG16 - AlexNet - CaffeNet - GoogLeNet - ResNet50 - ResNet101 - ResNet152"
	exit
	;;
esac

../../caffe/build/tools/caffe train --solver=${solver} --weights=${weights} 2>&1 | tee ${log}