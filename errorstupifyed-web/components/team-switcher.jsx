"use client"

import * as React from "react"
import { ChevronsUpDown, Plus, TrendingUp, Utensils } from "lucide-react"

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuShortcut,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  useSidebar,
} from "@/components/ui/sidebar"

export function TeamSwitcher({
  teams
}) {
  const { isMobile } = useSidebar()



  return (
    <SidebarMenu>
      <SidebarMenuItem>
       
            <div className="flex items-center gap-2 mt-2 data-[state=open]:bg-sidebar-accent data-[state=open]:text-sidebar-accent-foreground">
              <div
                className="bg-primary text-sidebar-primary-foreground flex aspect-square size-8 items-center justify-center rounded-lg">
                <TrendingUp className="size-4" />
              </div>
              <div className="grid flex-1 text-left text-sm leading-tight">
                <span className="truncate font-medium">Cognify Sales Analyzer</span>
                <span className="truncate text-xs">Error Stupifyed</span>
              </div>
             
            </div>
        
      </SidebarMenuItem>
    </SidebarMenu>
  );
}
