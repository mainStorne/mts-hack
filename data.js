/* Данные питча CompactFactExtraction. Метрики — val/test, оценщики v1/v2.1 кейса MTS MWS. */
window.MWS_DATA={
  labels:[],
  series:[],
  line:{yMin:0,yMax:1,yLabel:"",xLabel:""},
  barsDay1:{labels:["F1","Precision","Grounding"],series:[{name:"Бейзлайн glm-5.3",color:"#bbc1c7",values:[0.22,0.21,0.68]},{name:"GLiNER + LoRA",color:"#ff0032",values:[0.54,0.71,0.86]}],yMax:1,yLabel:"Значение метрики · оценщик v1"},
  barsDay2:{labels:["F1","Temporal","State"],series:[{name:"Бейзлайн glm-5.3 (v2.1)",color:"#bbc1c7",values:[0.356,0.471,0.535]},{name:"Gemma3 + LoRA",color:"#ff0032",values:[0.576,0.448,0.544]}],yMax:1,yLabel:"Значение метрики · val, оценщик v2.1"},
  scatter:{xMax:1,yMax:1,xLabel:"",yLabel:"",groups:[]},
  network:{nodes:[],edges:[],route:[]},
  architecture:{
    frame:{x:355,y:40,w:310,h:432,label:"Параллельные ветки"},
    nodes:[
      {id:"asr",x:30,y:205,w:205,h:100,title:"ASR-транскрипт",subtitle:"оператор и клиент"},
      {id:"gliner",x:395,y:85,w:230,h:100,title:"GLiNER2 + LoRA",subtitle:"факты-кандидаты со спанами"},
      {id:"gemma",x:395,y:330,w:230,h:100,title:"Gemma3 + LoRA",subtitle:"«одиночки» — вне GLiNER"},
      {id:"lr",x:815,y:205,w:260,h:100,title:"Логистическая регрессия",subtitle:"отбор фактов по признакам",active:true},
      {id:"out",x:1195,y:205,w:185,h:100,title:"Компактные факты",subtitle:"готовый ответ"}
    ],
    edges:[
      {from:"asr",to:"gliner",label:"реплики"},
      {from:"asr",to:"gemma",label:"реплики"},
      {from:"gliner",to:"lr",label:"кандидаты",active:true},
      {from:"gemma",to:"lr",label:"одиночки",active:true},
      {from:"lr",to:"out",label:"отобранные факты",active:true}
    ]
  }
};
