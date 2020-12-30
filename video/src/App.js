import React, { useState } from 'react';
//import camera from './components/camera';
import axios from "axios";
import './App.css';


function App() {
const [playing, setPlaying] = useState(false);

const HEIGHT = 500;
const WIDTH = 500;

const startVideo = () => {
setPlaying(true);
//navigator.getUserMedia(
//{
//video: true,
//},
//(stream) => {
//let video = document.getElementsByClassName('app__videoFeed')[0];
//if (video) {
//video.srcObject = stream;
//}
//},
//(err) => console.error(err)
//);

    axios({
      method: "get",
      url: "/video_feed",
//      data: formData,
      config: { headers: { "Content-Type": "multipart/form-data" } }
    }).then((response) => {
      this.setState({
        receiveDate: response.data,
        isLoad: !this.state.isLoad
      });
    });


};

const stopVideo = () => {
setPlaying(false);
let video = document.getElementsByClassName('app__videoFeed')[0];
video.srcObject.getTracks()[0].stop();
};





return (
    <div className="app">
    <div className="app__container">
           <video
            height={HEIGHT}
            width={WIDTH}
            muted
            autoPlay
            className="app__videoFeed"
    ></video>
    </div>
    <div className="app__input">
    {playing ? (
    <button onClick={stopVideo}>Stop</button>
    ) : (
    <button onClick={startVideo}>Start</button>
    )}
    </div>
    </div>
    );
    }




export default App;