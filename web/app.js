const samples = [
  {id:'demo-1',type:'Asfalt bozukluğu',status:'Örnek',confidence:.91,lat:41.0121,lng:28.9762,time:'2026-10-27T09:10:00'},
  {id:'demo-2',type:'Yerde çöp',status:'Örnek',confidence:.87,lat:41.0152,lng:28.9791,time:'2026-10-27T09:24:00'},
  {id:'demo-3',type:'Konteyner',status:'Örnek',confidence:.84,lat:41.0098,lng:28.9818,time:'2026-10-27T09:42:00'},
  {id:'demo-4',type:'Moloz',status:'Örnek',confidence:.79,lat:41.018,lng:28.973,time:'2026-10-27T10:05:00'},
  {id:'demo-5',type:'Sokak işgali',status:'Örnek',confidence:.76,lat:41.014,lng:28.984,time:'2026-10-27T10:16:00'},
  {id:'demo-6',type:'Pano',status:'Örnek',confidence:.68,lat:41.011,lng:28.970,time:'2026-10-27T10:28:00'}
];
const events = new Map();
const $ = id => document.getElementById(id);
let map = null, markerLayer = null;
if (window.L) {
  map=L.map('map').setView([41.013,28.978],14);
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap'}).addTo(map);
  markerLayer=L.layerGroup().addTo(map);
} else $('map').textContent='Harita kütüphanesi yüklenemedi; olay listesi kullanılabilir.';
const score = n => typeof n === 'number' ? n.toFixed(3) : 'bilinmiyor';
const containerState = name => ({empty:'Boş',full:'Atık var, taşmamış',overflowing:'Taşmış',unknown:'Durum belirsiz'}[name]||name);
const labels = {road_damage:'Yol hasarı adayı',litter_candidate:'Atık nesnesi adayı',scene:'Sahne gözlemi',lane_marking:'Yol işareti adayı',container:'Konteyner durum adayı'};
function addText(parent,tag,text) { const el=document.createElement(tag);el.textContent=text;parent.appendChild(el);return el; }
function records() {
  return [...($('showDemo').checked?samples:[]), ...events.values()].map(e=>e.event_id?{
    id:e.event_id,type:labels[e.event_type]||e.event_type,status:'İnceleme gerekli',confidence:e.confidence,
    lat:e.latitude,lng:e.longitude,time:e.analyzed_at||'',size:`${e.class_name} · ${e.observations} gözlem · video ${e.start_time.toFixed(2)}–${e.end_time.toFixed(2)} sn`,
    location:e.location_source
  }:e);
}
function updateOptions() {
  for (const [id,key] of [['type','type'],['status','status']]) {
    const el=$(id),value=el.value;el.replaceChildren(new Option(id==='type'?'Tüm türler':'Tüm durumlar',''));
    [...new Set(records().map(x=>x[key]))].forEach(x=>el.add(new Option(x,x)));
    if ([...el.options].some(o=>o.value===value)) el.value=value;
  }
}
function render() {
  if(markerLayer)markerLayer.clearLayers();$('cards').replaceChildren();
  const filtered=records().filter(x=>(!$('type').value||x.type===$('type').value)&&(!$('status').value||x.status===$('status').value)&&(!$('date').value||x.time.startsWith($('date').value)));
  $('count').textContent=`${filtered.length} kayıt`;
  if(!filtered.length)addText($('cards'),'p','Henüz analiz olayı yok. Örnekleri görmek için örnek kayıt kutusunu seçin.');
  for(const x of filtered) {
    const card=document.createElement('article');card.className='card';
    addText(card,'h3',`${x.type} · ${x.status}`);
    addText(card,'div',`Model skoru: ${score(x.confidence)} · ${x.size||'Örnek kayıt'}`);
    if(Number.isFinite(x.lat)&&Number.isFinite(x.lng)) {
      addText(card,'div',`Kamera GPS tahmini: ${x.lat.toFixed(5)}, ${x.lng.toFixed(5)}`);
      if(map){const popup=document.createElement('div');popup.textContent=`${x.type} · ${x.status}`;
        const marker=L.marker([x.lat,x.lng]).addTo(markerLayer).bindPopup(popup);
        card.onclick=()=>{map.setView([x.lat,x.lng],17);marker.openPopup();};}
    } else addText(card,'div','Konum bilinmiyor; harita işareti üretilmedi.');
    $('cards').appendChild(card);
  }
}
for(const id of ['type','status','date'])$(id).addEventListener('change',render);
$('showDemo').onchange=()=>{updateOptions();render();};
$('reset').onclick=()=>{for(const id of ['type','status','date'])$(id).value='';render();};
async function request(url,options) {
  const response=await fetch(url,options);const data=await response.json();
  if(!response.ok)throw new Error(data.error||`HTTP ${response.status}`);return data;
}
function showImage(box,data) {
  if(!data.image_base64)return;
  const img=document.createElement('img');img.style.maxWidth='100%';img.alt='İnceleme gerektiren model sonuçları';
  img.src=`data:image/jpeg;base64,${data.image_base64}`;box.appendChild(img);
}
$('run').onclick=async()=>{
  const box=$('modelResult'),file=$('image').files[0];box.style.display='block';box.replaceChildren();
  if(!file){box.textContent='Önce görsel seçin.';return;}
  $('run').disabled=true;box.textContent='Görsel analiz ediliyor…';
  const form=new FormData();form.append('image',file);
  try {const data=await request('/api/analyze',{method:'POST',body:form});box.replaceChildren();
    for(const [key,label] of [['road_damage','Yol hasarı adayları'],['ground_litter','Atık nesnesi adayları'],['scene_segments','Sahne gözlemleri'],['lane_markings','Yol işareti analizi'],['containers','Konteyner durumları']])
      addText(box,'p',`${label}: ${(data[key]||[]).map(x=>key==='containers'?`${containerState(x.class_name)} (durum skoru ${score(x.state_confidence)}, tespit skoru ${score(x.confidence)})`:`${x.class_name} (skor ${score(x.confidence)})`).join(', ')||'Aday yok'}`);
    const moduleNames={road:'Yol hasarı',litter:'Atık',objects:'Kişi/araç bağlamı',scene:'Yol/kaldırım/pano/tabela',lane:'Yol işaretleri',container:'Konteyner tespiti',container_state:'Konteyner durumu'};
    for(const [key,title] of Object.entries(moduleNames))addText(box,'div',`${title}: ${data.model_status?.[key]?.enabled?'Aktif':'Kapalı'}${data.model_status?.[key]?.status?.startsWith('experimental')?' · Deneysel':''}`);
    if(data.model_status?.lane?.enabled===false)addText(box,'p','Yol işareti modeli sıfır doğrulama başarımı nedeniyle devre dışı.');
    addText(box,'p','Sahne modeli düşük başarımlı bir pilottur; normal yol/kaldırım/tabela gözlemleri ihlal değildir.');
    addText(box,'p',data.container_note||'');addText(box,'p','Model skorları doğruluk yüzdesi değildir. Gizlilik ve sorun adayları manuel incelenmelidir.');showImage(box,data);
  }catch(e){box.textContent=`Hata: ${e.message}`;}finally{$('run').disabled=false;}
};
$('runContainer').onclick=async()=>{
  const box=$('containerResult'),file=$('containerImage').files[0];box.style.display='block';
  if(!file){box.textContent='Önce görsel seçin.';return;}
  box.textContent='Konteyner modeli kontrol ediliyor…';const form=new FormData();form.append('image',file);
  try{const data=await request('/api/container-classify',{method:'POST',body:form});box.replaceChildren();
    addText(box,'p',data.note||'');for(const item of data.detections)addText(box,'p',`${containerState(item.class_name)} · durum skoru ${score(item.state_confidence)} · tespit skoru ${score(item.confidence)}`);showImage(box,data);
  }catch(e){box.textContent=`Hata: ${e.message}`;}
};
let currentJob=null;
$('cancelVideo').onclick=async()=>{if(currentJob)await request(`/api/jobs/${currentJob}/cancel`,{method:'POST'});};
$('runVideo').onclick=async()=>{
  const box=$('videoResult'),file=$('video').files[0];box.style.display='block';box.replaceChildren();
  if(!file){box.textContent='Önce video seçin.';return;}
  $('runVideo').disabled=true;$('cancelVideo').disabled=false;
  const form=new FormData();form.append('video',file);if($('gps').files[0])form.append('gps',$('gps').files[0]);
  try {
    box.textContent='Video yükleniyor…';let job=await request('/api/analyze-video',{method:'POST',body:form});currentJob=job.job_id;
    while(['queued','processing'].includes(job.state)) {
      box.textContent=`${job.state==='queued'?'Sırada':'İşleniyor'} · ${job.frames_done}/${job.frames_total||'?'} kare`;
      await new Promise(resolve=>setTimeout(resolve,1000));job=await request(`/api/jobs/${currentJob}`);
    }
    if(job.state==='failed')throw new Error(job.error||'Video işlenemedi');
    if(job.state==='cancelled'){box.textContent='İşlem iptal edildi; geçici dosyalar temizlendi.';return;}
    const data=job.result;box.replaceChildren();
    addText(box,'p',`${data.frames} kare · ${data.fps.toFixed(2)} FPS · ${data.events.length} takip adayı`);
    if(data.timing_conversion?.variable_intervals_detected)addText(box,'p','Videonun değişken kare süreleri sabit FPS çıktısına uyarlandı.');
    addText(box,'p',`Yol işareti: ${data.counts.lane_markings||0} · Konteyner: ${data.counts.containers||0}`);
    addText(box,'p',data.privacy_note);addText(box,'p',data.count_note);
    const video=document.createElement('video');video.controls=true;video.playsInline=true;video.preload='metadata';video.style.maxWidth='100%';video.src=data.preview_url;box.appendChild(video);
    addText(box,'p','Bu önizleme yalnızca yerel oturumunuza açıktır. Yüz/plaka maskelemesini kontrol edin; eksikse çıktıyı onaylamayın.');
    const review=addText(box,'button','Gizlilik incelemesini yaptım');const jobId=currentJob;
    review.onclick=async()=>{try{await request(`/api/jobs/${jobId}/review`,{method:'POST'});review.disabled=true;review.textContent='Yerel inceleme kaydedildi';}catch(e){review.textContent=e.message;}};
    const analyzed=new Date().toISOString();
    for(const e of data.events)events.set(`${jobId}-${e.event_id}`,{...e,event_id:`${jobId}-${e.event_id}`,analyzed_at:analyzed});
    updateOptions();render();
  }catch(e){box.textContent=`Hata: ${e.message}`;}
  finally{$('runVideo').disabled=false;$('cancelVideo').disabled=true;currentJob=null;}
};
updateOptions();render();
