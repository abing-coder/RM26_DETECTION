#include "ArmorBox.h"

ArmorBox::ArmorBox(){
	armorNum = 0;
    armorVertices_.resize(4);
	type = SMALL_ARMOR;
	center = cv::Point2f();
}

ArmorBox::~ArmorBox(){}

// 提供给pnp解算和装甲板图像显示时用，避免提前调用增加耗时
void ArmorBox::setArmor(){
    if (aleadySetArmor) return;
    aleadySetArmor = true;
    armorVertices_.resize(4);
    // setArmorVertices_(l_light, r_light, *this);
    // set armor center
    // center = crossPointof(armorVertices_[0], armorVertices_[1], armorVertices_[2], armorVertices_[3]);
}
