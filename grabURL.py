# -*- coding: utf-8 -*-
"""parseUrls.py  A regular expression that detects HTTP urls.
Danny Yoo (dyoo@hkn.eecs.berkeley.edu)
This is only a small sample of tchrist's very nice tutorial on
regular expressions.  See:
    http://www.perl.com/doc/FMTEYEWTK/regexps.html
for more details.
Note: this properly detects strings like "http://python.org.", with a
period at the end of the string."""

import re

def grabUrls(text):
    """Given a text string, returns all the urls we can find in it."""
    urls = '(?: %s)' % '|'.join("""http https telnet gopher file waisftp""".split())
    ltrs = r'\w'
    gunk = r'/#~:.?+=&%@!\-'
    punc = r'.:?\-'
    any = "%(ltrs)s%(gunk)s%(punc)s" % { 'ltrs' : ltrs, 'gunk' : gunk, 'punc' : punc }

    url = r"""
        \b                            # start at word boundary
            %(urls)s    :             # need resource and a colon
            [%(any)s]  +?             # followed by one or more
                                      #  of any valid character, but
                                      #  be conservative and take only
                                      #  what you need to....
        (?=                           # look-ahead non-consumptive assertion
                [%(punc)s]*           # either 0 or more punctuation
                (?:   [^%(any)s]      #  followed by a non-url char
                    |                 #   or end of the string
                      $
                )
        )
        """ % {'urls' : urls, 'any' : any, 'punc' : punc }

    url_re = re.compile(url, re.VERBOSE | re.MULTILINE)
    found_urls = url_re.findall(text)

    return found_urls

if __name__ == '__main__':
    sample = 'Code for paper (https://guzpenha.github.io/MANtIS/) is available.'
    match = grabUrls(sample)
    print("Here's what we found: '%s'" % match[0])