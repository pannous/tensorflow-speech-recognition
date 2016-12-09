#!/usr/bin/env swift
import AppKit

var tts=NSSpeechSynthesizer.init(voice:"com.apple.speech.synthesis.voice.Vicki")!
var text=""
var max=CommandLine.arguments.count-1;
for s in CommandLine.arguments[1 ... max] {
  text+=s+" "
}
extension NSString {
    func split(pattern: String) -> [String] {return self.components(separatedBy:pattern);    }
    var strip:String { return self.trimmingCharacters(in:NSCharacterSet.whitespacesAndNewlines)}
    func replace(pattern: String,with:String)->String{return self.replacingOccurrences(of:pattern,with:with);}
}

var phon:String=tts.phonemes(from:text)
print(phon)
