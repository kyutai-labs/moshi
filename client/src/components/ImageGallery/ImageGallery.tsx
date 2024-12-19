
import { useState } from "react";
import { Button } from "../Button/Button";

// Natural images
import img1 from "/assets/images/demo/image1.jpg";
import img2 from "/assets/images/demo/image2.jpg";
import img3 from "/assets/images/demo/image3.jpg";
import img4 from "/assets/images/demo/image4.jpg";
import img5 from "/assets/images/demo/image5.jpg";
import img6 from "/assets/images/demo/image6.jpg";
import img7 from "/assets/images/demo/image7.jpg";
import img8 from "/assets/images/demo/image8.jpg";
import img9 from "/assets/images/demo/image9.jpg";
import img10 from "/assets/images/demo/image10.jpg";
import img11 from "/assets/images/demo/image11.jpg";
import img12 from "/assets/images/demo/image12.jpg";
import img13 from "/assets/images/demo/image13.jpg";
import img14 from "/assets/images/demo/image14.jpg";
import img15 from "/assets/images/demo/image15.jpg";
import img16 from "/assets/images/demo/image16.jpg";
import img17 from "/assets/images/demo/image17.jpg";
import img18 from "/assets/images/demo/image18.jpg";
import img19 from "/assets/images/demo/image19.jpg";
import img20 from "/assets/images/demo/image20.jpg";

const images = [
    img1,
    img2,
    img3,
    img4,
    img5,
    img6,
    img7,
    img8,
    img9,
    img10,
    img11,
    img12,
    img13,
    img14,
    img15,
    img16,
    img17,
    img18,
    img19,
    img20,
]

var images_order: number[] = [];
for (let i = 0; i < images.length; i++) {
    images_order.push(i)
}

type ImageGalleryProps = React.InputHTMLAttributes<HTMLInputElement> & {
    // Properties for the ImageGallery
    paramsSetter: Function;
    clickAction: Function;
    size: number;
    numImages: number;
}


type ImageItemProps = React.InputHTMLAttributes<HTMLInputElement> & {
    // Properties for a single item in the ImageGallery
    // Two actions:
    // paramsSetter sets the chosen image url into the model params
    // clickAction then starts the conversation
    paramsSetter: Function;
    clickAction: Function;
    size: number;
    imageUrl: string;
}


function ImageSelect(props: ImageItemProps) {
    // Represents a single image in the gallery
    const [isHover, setIsHover] = useState(false);

    const handleMouseEnter = () => {
        setIsHover(true);
    };
    const handleMouseLeave = () => {
        setIsHover(false);
    };
    let bordercolor = isHover ? "#f7a319" : "black";
    let bgalpha = isHover ? 0.05 : 0.6;
    let textalpha = isHover ? 1.0 : 0.0
    let label = isHover ? "Connect" : "X";
    let style = {
        width: props.size,
        height: props.size,
        background: `url(${props.imageUrl})`,
        backgroundSize: "100% 100%",
        border: `3px solid ${bordercolor}`,
        margin: "2px",
        padding: "0px",
        color: `rgba(255, 255, 255, ${textalpha})`,
        boxShadow: `inset 0 0 0 1000px rgba(0,0,0,${bgalpha})`,
        textShadow: `2px 2px 2px rgba(0, 0, 0, ${textalpha})`
    };
    return (
        <button style={style} onMouseEnter={handleMouseEnter} onMouseLeave={handleMouseLeave}
            onClick={async () => { await props.paramsSetter(props.imageUrl); props.clickAction() }
            } > {label}</button >
    );
}


const shuffle = (array: number[]) => {
    return array.sort(() => Math.random() - 0.5);
};

export const ImageGallery = (props: ImageGalleryProps) => {
    const [ordering, SetOrdering] = useState(images_order);

    function handleShuffle() {
        SetOrdering(shuffle([...ordering]));
    }

    // Image Gallery widget (random subset)
    const steps = [];
    for (let i = 0; i < props.numImages; i++) {
        steps.push(<ImageSelect
            key={"natural_" + ordering[i]}
            imageUrl={images[ordering[i]]} {...props}></ImageSelect >);
    }

    return (
        <div>
            <div className="flex justify-center items-center  p-2 flex-center" style={{ marginRight: "12%", marginLeft: "12%" }}>
                <span style={{ display: "flex", flex: 1 }}></span>
                <Button onClick={handleShuffle} style={{ display: "flex" }}>
                    🔄
                </Button>
            </div>
            <div className="imageGallery" >{steps}</div>
        </div >)
        ;
};
