import AppKit
// me:~$ say Adamic
// me:~$ say Adamec
var tts=NSSpeechSynthesizer.init(voice:"com.apple.speech.synthesis.voice.Vicki")
var text="Hallo"
for s in Process.arguments {
  text=s
}
extension NSString {
    func split(pattern: String) -> [AnyObject] {return self.componentsSeparatedByString(pattern);    }
    var strip:String { return self.stringByTrimmingCharactersInSet(NSCharacterSet.whitespaceAndNewlineCharacterSet())}
    func replace(pattern: String,with:String)->String{return self.stringByReplacingOccurrencesOfString(pattern,withString:with);}
    init(contentsOfFile:path){
	let text : String; do {text = try String(contentsOfFile: fileLocation);} catch {text="ERROR";};return text; }
}

// tts.delegate=MyDelegate()
// tts.startSpeakingString("Hallo")

println(tts.phonemesFromText(text))


var path="/Users/me/data/base/lang/english.index"
let content = NSString(contentsOfFile:path) //, encoding: NSUTF8StringEncoding, error: nil) as String
var all=content.split("\n")
// let content = NSArray(contentsOfFile:path)

for obj: AnyObject in all {
  var word=obj as String
  var phoneme=tts.phonemesFromText(word).replace(" ~",with:"~") as String
  // if var rawStandard = obj as? NSDictionary {
  // println(obj)
  println(word.strip+"\t"+phoneme)
}

// println(content)
// cat ~/data/base/words/english.index| while read w;do echo "$w"; say "$w" -o "spoken_words/$w.m4a";done


// @objc class MyDelegate : NSObject,NSSpeechSynthesizerDelegate{// ,NSObjectProtocol
//   // override _ sender:
//    func speechSynthesizer(sender: NSSpeechSynthesizer, willSpeakPhoneme phoneme: Int16){
//     print("Doesn't fucking work why not")
//   }
//    func run(){
//      var tts=NSSpeechSynthesizer.init(voice:"com.apple.speech.synthesis.voice.Vicki")
//      tts.delegate=self
//      tts.startSpeakingString(text)
//    }
// }
// MyDelegate().run()
// sleep(2);

