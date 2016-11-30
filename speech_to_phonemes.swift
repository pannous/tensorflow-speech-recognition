#!/usr/bin/env swift
import AppKit
// me:~$ say Adamic
// me:~$ say Adamec
print("GO!")

var tts=NSSpeechSynthesizer.init(voice:"com.apple.speech.synthesis.voice.Vicki")!
var text="Hallo"
for s in CommandLine.arguments {
  text=s
}
extension NSString {
    func split(pattern: String) -> [String] {return self.components(separatedBy:pattern);    }
    var strip:String { return self.trimmingCharacters(in:NSCharacterSet.whitespacesAndNewlines)}
    func replace(pattern: String,with:String)->String{return self.replacingOccurrences(of:pattern,with:with);}
}

// tts.delegate=MyDelegate()
// tts.startSpeakingString("Hallo")

print(tts.phonemes(from:text))


var path="/Users/me/data/base/lang/english.index"
let data: NSData = NSData(contentsOfFile: path)!
// let content = NSString(contentsOfFile:path, encoding: NSUTF8StringEncoding, error: nil) as String
let content = try String(contentsOf:URL(fileURLWithPath:path)) 
var all=content.split(pattern:"\n")
// let content = NSArray(contentsOfFile:path)

for obj: String in all {
  var word=obj as String
  var phoneme=tts.phonemes(from:word).replace(pattern:" ~",with:"~") as String
  // if var rawStandard = obj as? NSDictionary {
  // print(obj)
  print(word.strip+"\t"+phoneme)
}

// print(content)
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

