"use client"

import { ChevronRight } from "lucide-react";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

import {
  SidebarGroup,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
} from "@/components/ui/sidebar"
import Cookies from "js-cookie"; 
import { useRouter } from "next/navigation";


export function NavMain({
  items
}) 

{
  const router = useRouter();
  return (
    
    <SidebarGroup>
      <SidebarGroupLabel>Manage</SidebarGroupLabel>
      <SidebarMenu>
        {items.map((item) => (
          <Collapsible
            key={item.title}
            asChild
            defaultOpen={item.isActive}
          
            
            className={item.isActive? "p-1" : "p-1"}>
            <SidebarMenuItem>
            
                <SidebarMenuButton tooltip={item.title} onClick={() => {
                  if(item.title === "Logout") {
                    Cookies.remove("uid");
                    router.push("/")
                  }
                  
                  router.push(item.url)
                }
                }>
                  {item.icon && <item.icon />}
                  <span>{item.title}</span>
                 
                </SidebarMenuButton>
             
             
            </SidebarMenuItem>
          </Collapsible>
        ))}
      </SidebarMenu>
    </SidebarGroup>
  );
}
